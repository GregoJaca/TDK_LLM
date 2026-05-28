import os
import yaml
import pickle
import torch
import numpy as np
import hashlib
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.utils import ensure_dir

# Custom LayerNorm module for validation if model does not have one
class SimpleLayerNorm(torch.nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(normalized_shape, device=device, dtype=dtype))
        self.bias = torch.nn.Parameter(torch.zeros(normalized_shape, device=device, dtype=dtype))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(var + self.eps) * self.weight + self.bias

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def is_rms_norm(module):
    name = module.__class__.__name__.lower()
    return "rms" in name or "rmsnorm" in name

def main():
    config_path = "jacobian_config.yaml"
    if not os.path.exists(config_path):
        print(f"Error: {config_path} not found.")
        return
        
    config = load_config(config_path)
    model_config = config.get("model", {})
    exp_config = config.get("experiment", {})
    norm_config = config.get("normalization", {})
    
    use_rms = norm_config.get("use_rms", True)
    use_layernorm = norm_config.get("use_layernorm", True)
    results_dir = norm_config.get("results_dir", "./results_normalization")
    radii = norm_config.get("radii", [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0])
    num_sampled_tokens = norm_config.get("num_sampled_tokens", 50)
    
    ensure_dir(results_dir)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = model_config.get("name", "microsoft/Phi-4-mini-instruct")
    local_dir = model_config.get("local_dir", ".")
    
    dtype_str = model_config.get("torch_dtype", "auto")
    if dtype_str == "float32":
        torch_dtype = torch.float32
    elif dtype_str == "bfloat16":
        torch_dtype = torch.bfloat16
    elif dtype_str == "float16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print(f"Loading tokenizer & model: {model_name} on {device} ({torch_dtype})...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=local_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch_dtype,
        cache_dir=local_dir
    )
    model.to(device)
    model.eval()
    
    num_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    
    # We will hook into the post_attention_layernorm layers
    layer_inputs = {}
    hooks = []
    
    def get_hook(l_idx):
        def hook(module, input, output):
            # Input to the normalization layer is input[0]
            # Shape is typically [batch_size, seq_len, hidden_size]
            layer_inputs[l_idx] = input[0].detach().clone()
        return hook
        
    for i in range(num_layers):
        # Phi-4 / Qwen architecture uses post_attention_layernorm
        norm_layer = model.model.layers[i].post_attention_layernorm
        h = norm_layer.register_forward_hook(get_hook(i))
        hooks.append(h)
        
    # Get the first prompt from the config
    prompts_cfg = exp_config.get("prompts", [])
    if not prompts_cfg:
        print("Error: No prompts found in config.")
        return
        
    prompt_id = exp_config.get("setups", [{}])[0].get("prompt_id", "long_prompts")
    prompt_texts = None
    for p in prompts_cfg:
        if p["id"] == prompt_id:
            prompt_texts = p.get("texts", [p.get("text", "")])
            break
            
    if not prompt_texts:
        print(f"Error: Prompt ID '{prompt_id}' not found.")
        return
        
    prompt_text = prompt_texts[0]
    prompt_hash = hashlib.md5(prompt_text.encode('utf-8')).hexdigest()[:8]
    print(f"Running forward pass on prompt (hash: {prompt_hash})...")
    
    encoded = tokenizer(prompt_text, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    
    with torch.no_grad():
        _ = model(input_ids=input_ids, attention_mask=attention_mask)
        
    # Remove hooks
    for h in hooks:
        h.remove()
        
    print(f"Captured hidden states for {len(layer_inputs)} layers.")
    
    # Process and sample tokens
    # layer_inputs[0] shape: [1, seq_len, hidden_size]
    seq_len = layer_inputs[0].shape[1]
    
    # Sample tokens evenly across the sequence
    sampled_indices = np.linspace(0, seq_len - 1, min(num_sampled_tokens, seq_len), dtype=int)
    print(f"Sampling {len(sampled_indices)} tokens out of {seq_len}...")
    
    results = {
        "model_name": model_name,
        "prompt_hash": prompt_hash,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "radii": radii,
        "sampled_indices": sampled_indices.tolist(),
        "data": []
    }
    
    # Helper functions to compute JVPs in batch
    def compute_jvp_batch(norm_fn, x, delta_x_batch):
        # x: [D], delta_x_batch: [num_radii, D]
        def single_jvp(d_x):
            _, out_jvp = torch.func.jvp(norm_fn, (x,), (d_x,))
            return out_jvp
        return torch.func.vmap(single_jvp)(delta_x_batch)

    # Let's run the experiments
    for layer_idx in range(num_layers):
        print(f"Processing Layer {layer_idx + 1}/{num_layers}...")
        
        # Unnormalized activations for this layer
        # shape: [seq_len, hidden_size]
        x_all = layer_inputs[layer_idx].squeeze(0).to(torch.float32)
        
        # Get actual model RMSNorm layer
        actual_rms_module = model.model.layers[layer_idx].post_attention_layernorm
        actual_eps = getattr(actual_rms_module, "variance_epsilon", 1e-6)
        actual_gamma = getattr(actual_rms_module, "weight", None)
        
        # If we need LayerNorm, we instantiate a validation layer matching the config
        # with eps=1e-5 and weight=1.0, bias=0.0 (pure LN)
        validation_ln = SimpleLayerNorm(hidden_size, eps=1e-5, device=device, dtype=torch.float32)
        
        for token_idx in sampled_indices:
            x = x_all[token_idx].to(device) # [D]
            
            x_norm_l2 = torch.norm(x, p=2).item()
            
            # LayerNorm quantities
            x_mean = x.mean().item()
            x_tilde = x - x_mean
            x_std = torch.sqrt(x_tilde.pow(2).mean() + 1e-5).item()
            
            # Setup normalization configurations to test
            norm_configs = []
            if use_rms:
                # 1. Actual model RMSNorm (includes gamma, eps)
                norm_configs.append({
                    "name": "rms_actual",
                    "module": actual_rms_module,
                    "eps": actual_eps,
                    "gamma": actual_gamma,
                    "type": "rms"
                })
                # 2. Pure RMSNorm (gamma = 1, eps = 1e-6)
                pure_rms_module = lambda v: v / torch.sqrt(v.pow(2).mean(-1, keepdim=True) + 1e-6)
                norm_configs.append({
                    "name": "rms_pure",
                    "module": pure_rms_module,
                    "eps": 1e-6,
                    "gamma": None,
                    "type": "rms"
                })
                
            if use_layernorm:
                # 3. Pure LayerNorm (gamma = 1, bias = 0)
                norm_configs.append({
                    "name": "layernorm_pure",
                    "module": validation_ln,
                    "eps": 1e-5,
                    "gamma": None,
                    "type": "layernorm"
                })
                
            for norm_cfg in norm_configs:
                norm_name = norm_cfg["name"]
                norm_module = norm_cfg["module"]
                eps_val = norm_cfg["eps"]
                gamma_val = norm_cfg["gamma"]
                norm_type = norm_cfg["type"]
                
                # Define functional wrapper for JVP
                def norm_fn(v):
                    params = list(norm_module.parameters()) if isinstance(norm_module, torch.nn.Module) else []
                    m_dtype = params[0].dtype if params else torch.float32
                    out = norm_module(v.unsqueeze(0).unsqueeze(0).to(m_dtype))
                    return out.to(torch.float32).view(-1)
                
                # Precompute baseline output
                with torch.no_grad():
                    y = norm_fn(x)
                
                # Create perturbations of different directions
                # 1. Radial direction (along x)
                v_radial = x / max(x_norm_l2, 1e-12)
                
                # 2. Orthogonal direction (perpendicular to x, and for LN, perpendicular to 1)
                # Draw random Gaussian vector
                v_orth = torch.randn(hidden_size, device=device, dtype=torch.float32)
                if norm_type == "rms":
                    # Project orthogonal to x
                    v_orth = v_orth - (torch.dot(x, v_orth) / max(x_norm_l2**2, 1e-12)) * x
                else:
                    # For LN, project orthogonal to both 1 and x_tilde
                    ones = torch.ones_like(x)
                    # First project orthogonal to 1
                    v_orth = v_orth - (torch.dot(ones, v_orth) / hidden_size) * ones
                    # Then project orthogonal to x_tilde
                    x_tilde_norm_sq = torch.dot(x_tilde, x_tilde)
                    v_orth = v_orth - (torch.dot(x_tilde, v_orth) / max(x_tilde_norm_sq, 1e-12)) * x_tilde
                    
                v_orth = v_orth / torch.norm(v_orth, p=2)
                
                # Generate batched perturbation magnitudes (radii)
                radii_tensor = torch.tensor(radii, device=device, dtype=torch.float32).unsqueeze(-1) # [num_radii, 1]
                
                delta_x_radial_batch = radii_tensor * v_radial.unsqueeze(0) # [num_radii, D]
                delta_x_orth_batch = radii_tensor * v_orth.unsqueeze(0) # [num_radii, D]
                
                for pert_name, delta_x_batch in [("radial", delta_x_radial_batch), ("orthogonal", delta_x_orth_batch)]:
                    # Compute JVPs in parallel using vmap
                    delta_y_jvp_batch = compute_jvp_batch(norm_fn, x, delta_x_batch) # [num_radii, D]
                    
                    # Process each radius
                    for r_idx, radius in enumerate(radii):
                        delta_x = delta_x_batch[r_idx]
                        delta_y_jvp = delta_y_jvp_batch[r_idx]
                        
                        # Option 3: Empirical Finite Difference
                        with torch.no_grad():
                            y_perturbed = norm_fn(x + delta_x)
                            delta_y_emp = y_perturbed - y
                            
                        # Option 1: Analytical formula evaluation
                        if norm_type == "rms":
                            S_rms = np.sqrt(x_norm_l2**2 / hidden_size + eps_val)
                            delta_y_theory = (1.0 / S_rms) * (delta_x - (torch.dot(x, delta_x) / (hidden_size * S_rms**2)) * x)
                        else: # layernorm
                            # LayerNorm standard deviation
                            S_ln = x_std
                            # Project delta_x orthogonal to 1
                            dx_mean = delta_x.mean()
                            dx_centered = delta_x - dx_mean
                            # Compute the projection term along x_tilde
                            x_tilde_dot_dx = torch.dot(x_tilde, delta_x)
                            delta_y_theory = (1.0 / S_ln) * (dx_centered - (x_tilde_dot_dx / (hidden_size * S_ln**2)) * x_tilde)
                            
                        if gamma_val is not None:
                            delta_y_theory_weighted = delta_y_theory * gamma_val
                        else:
                            delta_y_theory_weighted = delta_y_theory
                            
                        # Compute scalar metrics to save disk space
                        emp_norm = torch.norm(delta_y_emp, p=2).item()
                        jvp_norm = torch.norm(delta_y_jvp, p=2).item()
                        theory_norm = torch.norm(delta_y_theory, p=2).item()
                        theory_weighted_norm = torch.norm(delta_y_theory_weighted, p=2).item()
                        
                        # Cosine similarities
                        def cos_sim(v1, v2):
                            n1 = torch.norm(v1, p=2)
                            n2 = torch.norm(v2, p=2)
                            if n1 == 0 or n2 == 0:
                                return 0.0
                            return (torch.dot(v1, v2) / (n1 * n2)).item()
                            
                        # For LN, we compare alignment with x_tilde (the direction that is projected out)
                        align_target = x_tilde if norm_type == "layernorm" else x
                        
                        cos_sim_emp_x = cos_sim(delta_y_emp, align_target)
                        cos_sim_jvp_x = cos_sim(delta_y_jvp, align_target)
                        cos_sim_theory_x = cos_sim(delta_y_theory_weighted, align_target)
                        cos_sim_emp_jvp = cos_sim(delta_y_emp, delta_y_jvp)
                        
                        # Absolute difference in norms (verification error)
                        norm_err_theory = abs(emp_norm - theory_weighted_norm)
                        norm_err_jvp = abs(emp_norm - jvp_norm)
                        
                        results["data"].append({
                            "layer": layer_idx,
                            "token_idx": int(token_idx),
                            "norm_name": norm_name,
                            "norm_type": norm_type,
                            "pert_type": pert_name,
                            "radius": float(radius),
                            "x_norm_l2": float(x_norm_l2),
                            "x_std": float(x_std),
                            "emp_norm": float(emp_norm),
                            "jvp_norm": float(jvp_norm),
                            "theory_norm": float(theory_norm),
                            "theory_weighted_norm": float(theory_weighted_norm),
                            "cos_sim_emp_x": float(cos_sim_emp_x),
                            "cos_sim_jvp_x": float(cos_sim_jvp_x),
                            "cos_sim_theory_x": float(cos_sim_theory_x),
                            "cos_sim_emp_jvp": float(cos_sim_emp_jvp),
                            "norm_err_theory": float(norm_err_theory),
                            "norm_err_jvp": float(norm_err_jvp)
                        })

    raw_data_path = os.path.join(results_dir, "raw_normalization_data.pkl")
    with open(raw_data_path, "wb") as f:
        pickle.dump(results, f)
        
    print(f"Raw data successfully saved to {raw_data_path}")

if __name__ == "__main__":
    main()
