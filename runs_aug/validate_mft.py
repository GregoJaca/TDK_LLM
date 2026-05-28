import os
import json
import glob
import yaml
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    config_path = "jacobian_config.yaml"
    if os.path.exists(config_path):
        config = load_config(config_path)
    else:
        print("Error: jacobian_config.yaml not found.")
        return
        
    model_config = config.get("model", {})
    exp_config = config.get("experiment", {})
    mft_config = config.get("mft_validation", {})
    
    model_name = model_config.get("name", "microsoft/Phi-4-mini-instruct")
    local_dir = model_config.get("local_dir", ".")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_results_dir = exp_config.get("results_dir", "./results_jacobians")
    
    metrics_filename = mft_config.get("metrics_filename", "mft_metrics.json")
    
    # 1. Scan for run directories that have both config.json and mlp_jacobian_measurements.json
    run_dirs = []
    config_files = glob.glob(os.path.join(base_results_dir, "*", "config.json"))
    for cfg_path in config_files:
        run_dir = os.path.dirname(cfg_path)
        jac_path = os.path.join(run_dir, "mlp_jacobian_measurements.json")
        if os.path.exists(jac_path):
            run_dirs.append(run_dir)
            
    if not run_dirs:
        print(f"No completed run directories found in {base_results_dir}.")
        return
        
    print(f"Found {len(run_dirs)} runs to validate. Loading model: {model_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=local_dir)
    
    dtype_str = model_config.get("torch_dtype", "auto")
    if dtype_str == "float32":
        torch_dtype = torch.float32
    elif dtype_str == "bfloat16":
        torch_dtype = torch.bfloat16
    elif dtype_str == "float16":
        torch_dtype = torch.float16
    else:
        torch_dtype = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float32 if torch_dtype == "float32" else (torch.bfloat16 if torch_dtype == "bfloat16" else (torch.float16 if torch_dtype == "float16" else "auto")),
        cache_dir=local_dir
    )
    model.to(device)
    model.eval()
    
    num_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    
    # 2. Iterate over each run and compute the MFT metrics
    for run_idx, run_dir in enumerate(run_dirs):
        print(f"\n[{run_idx+1}/{len(run_dirs)}] Processing: {os.path.basename(run_dir)}", flush=True)
        
        with open(os.path.join(run_dir, "config.json"), "r") as f:
            run_config = json.load(f)
            
        prompt_text = run_config.get("prompt_text")
        if not prompt_text:
            print("  Skipping because prompt_text is missing in config.json")
            continue
            
        with open(os.path.join(run_dir, "mlp_jacobian_measurements.json"), "r") as f:
            jac_data = json.load(f)
            
        # Hook into the post_attention_layernorm to capture x_norm
        layer_x_norms = {}
        
        def get_hook(l_idx):
            def hook(module, input, output):
                layer_x_norms[l_idx] = output.detach().clone().cpu()
            return hook
            
        hooks = []
        for i in range(num_layers):
            h = model.model.layers[i].post_attention_layernorm.register_forward_hook(get_hook(i))
            hooks.append(h)
            
        # Run forward pass to extract activations
        encoded = tokenizer(prompt_text, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        
        print("  Running forward pass to extract layer inputs...", flush=True)
        with torch.no_grad():
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
            
        for h in hooks:
            h.remove()
            
        mft_results = {
            "layers": {}
        }
        
        print("  Evaluating MFT assumptions layer-by-layer...", flush=True)
        for layer_idx in range(num_layers):
            # Extract MLP weights
            mlp = model.model.layers[layer_idx].mlp
            
            # Identify proj layers (using same logic as jacobian_measurements.py)
            if hasattr(mlp, 'gate_proj') and hasattr(mlp, 'up_proj'):
                W_gate = mlp.gate_proj.weight.data
                W_up = mlp.up_proj.weight.data
            elif hasattr(mlp, 'gate_up_proj'):
                W_gate_up = mlp.gate_up_proj.weight.data
                split_dim = W_gate_up.shape[0] // 2
                W_gate = W_gate_up[:split_dim, :]
                W_up = W_gate_up[split_dim:, :]
            else:
                linears = [m for m in mlp.modules() if isinstance(m, torch.nn.Linear)]
                if len(linears) >= 3:
                    W_gate = linears[0].weight.data
                    W_up = linears[1].weight.data
                elif len(linears) == 2:
                    W_gate_up = linears[0].weight.data
                    split_dim = W_gate_up.shape[0] // 2
                    W_gate = W_gate_up[:split_dim, :]
                    W_up = W_gate_up[split_dim:, :]
                else:
                    raise AttributeError(f"Could not automatically determine MLP weight matrices for {type(mlp)}")
                    
            if hasattr(mlp, 'down_proj'):
                W_down = mlp.down_proj.weight.data
            else:
                linears = [m for m in mlp.modules() if isinstance(m, torch.nn.Linear)]
                if len(linears) >= 2:
                    W_down = linears[-1].weight.data
                else:
                    raise AttributeError(f"Could not automatically determine MLP down_proj for {type(mlp)}")
                    
            # Move to float32 on device
            W_gate_32 = W_gate.to(device=device, dtype=torch.float32)
            W_up_32 = W_up.to(device=device, dtype=torch.float32)
            W_down_32 = W_down.to(device=device, dtype=torch.float32)
            
            # --- Assumption 2: Uniform Row Norms (Static Check) ---
            # W_gate and W_up are [d_ff, d]. We compute row-wise norms.
            # W_down is [d, d_ff]. We compute column-wise norms (dim=0).
            N_gate = (W_gate_32 ** 2).sum(dim=1)
            N_up = (W_up_32 ** 2).sum(dim=1)
            N_down = (W_down_32 ** 2).sum(dim=0)
            
            CV_gate = (torch.sqrt(N_gate).std() / torch.sqrt(N_gate).mean()).item()
            CV_up = (torch.sqrt(N_up).std() / torch.sqrt(N_up).mean()).item()
            CV_down = (torch.sqrt(N_down).std() / torch.sqrt(N_down).mean()).item()
            
            # --- Assumption 1: Vanishing Off-Diagonals (Dynamic Check) ---
            # Retrieve activation vectors S_x and D_x
            x_norm = layer_x_norms[layer_idx].to(device=device, dtype=torch.float32)
            if x_norm.dim() == 3:
                x_norm = x_norm.squeeze(0) # [seq_len, hidden_size]
                
            h_gate = torch.nn.functional.linear(x_norm, W_gate_32)
            h_up = torch.nn.functional.linear(x_norm, W_up_32)
            
            sig_h_gate = torch.sigmoid(h_gate)
            S_x = h_gate * sig_h_gate
            silu_prime_h_gate = sig_h_gate * (1 + h_gate * (1 - sig_h_gate))
            D_x = h_up * silu_prime_h_gate
            
            C_cross = (W_gate_32 * W_up_32).sum(dim=1)
            
            # Compute diagonal-only phase space expansion E_diag
            S_x2 = S_x ** 2
            D_x2 = D_x ** 2
            E_diag = (N_down * (S_x2 * N_up + D_x2 * N_gate + 2 * S_x * D_x * C_cross)).sum(dim=-1)
            lambda_diagonal = E_diag / hidden_size # [seq_len]
            
            # Load corresponding true Jacobian measurements
            layer_str = str(layer_idx)
            if layer_str not in jac_data["layers"]:
                # Try integer key matching
                matched_key = None
                for k in jac_data["layers"].keys():
                    if int(k) == layer_idx:
                        matched_key = k
                        break
                if matched_key:
                    layer_data = jac_data["layers"][matched_key]
                else:
                    print(f"  Warning: Layer {layer_idx} missing in true jacobian data. Skipping.")
                    continue
            else:
                layer_data = jac_data["layers"][layer_str]
                
            lambda_true = np.array(layer_data["lambda_true"])
            
            # Option B: Ratio of Sequence Averages
            mean_lambda_diagonal = lambda_diagonal.mean().item()
            mean_lambda_true = float(np.mean(lambda_true))
            R_option_b = mean_lambda_diagonal / mean_lambda_true if mean_lambda_true > 0 else 0.0
            
            # For the fan shading, we compute the token-wise ratio distribution R_t
            lambda_true_tensor = torch.tensor(lambda_true, device=device, dtype=torch.float32)
            R_t = lambda_diagonal / torch.clamp(lambda_true_tensor, min=1e-12)
            R_t_np = R_t.cpu().numpy()
            
            p10, p25, p75, p90 = np.percentile(R_t_np, [10, 25, 75, 90])
            median_ratio = np.median(R_t_np)
            
            mft_results["layers"][layer_idx] = {
                "CV_gate": CV_gate,
                "CV_up": CV_up,
                "CV_down": CV_down,
                "lambda_diag_mean": mean_lambda_diagonal,
                "lambda_true_mean": mean_lambda_true,
                "R_option_b": R_option_b,
                "R_t_p10": float(p10),
                "R_t_p25": float(p25),
                "R_t_p75": float(p75),
                "R_t_p90": float(p90),
                "R_t_median": float(median_ratio)
            }
            
            # Clear CUDA variables immediately to prevent OOM
            del W_gate_32, W_up_32, W_down_32, x_norm, h_gate, h_up, sig_h_gate, S_x, D_x, C_cross, E_diag, lambda_diagonal, lambda_true_tensor, R_t
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        # Save metrics inside the run directory to prevent filename collisions
        out_metrics_path = os.path.join(run_dir, metrics_filename)
        with open(out_metrics_path, "w") as f:
            json.dump(mft_results, f, indent=4)
        print(f"  Saved metrics to {out_metrics_path}", flush=True)

if __name__ == "__main__":
    main()
