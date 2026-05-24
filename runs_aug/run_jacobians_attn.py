import os
import json
import yaml
import torch
import time
import hashlib
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.jacobian_measurements import compute_attn_jacobian_metrics

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def main():
    config_path = "jacobian_config.yaml"
    if os.path.exists(config_path):
        config = load_config(config_path)
    else:
        print("Error: jacobian_config.yaml not found.")
        return
        
    model_config = config.get("model", {})
    exp_config = config.get("experiment", {})
    
    model_name = model_config.get("name", "microsoft/Phi-4-mini-instruct")
    local_dir = model_config.get("local_dir", ".")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_results_dir = exp_config.get("results_dir", "./results_jacobians_microsoft")
    ensure_dir(base_results_dir)
    
    # Retrieve experiment parameters
    N_list = exp_config.get("attn_seq_lengths", [20, 100, 1000])
    K = exp_config.get("attn_power_iterations", 20)
    max_N = max(N_list)
    
    print(f"Loading model: {model_name} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=local_dir)
    
    # Retrieve dtype configuration (default to auto/bfloat16 for eager attention stability)
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
        cache_dir=local_dir,
        attn_implementation="eager"
    )
    model.to(device)
    model.eval()
    
    prompts = {}
    for p in exp_config.get("prompts", []):
        if "texts" in p:
            prompts[p["id"]] = p["texts"]
        else:
            prompts[p["id"]] = [p["text"]]

    for setup in exp_config.get("setups", []):
        print(f"\n=== Running Attention Setup: {setup['name']} ===", flush=True)
        
        prompt_texts = prompts[setup["prompt_id"]]
        for prompt_idx, prompt_text in enumerate(prompt_texts):
            prompt_hash = hashlib.md5(prompt_text.encode('utf-8')).hexdigest()[:8]
            
            print(f"\n--- Prompt {prompt_idx + 1}/{len(prompt_texts)} ---", flush=True)
            
            encoded = tokenizer(prompt_text, return_tensors="pt")
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            
            seq_len = input_ids.shape[1]
            print(f"Tokenized prompt length: {seq_len} tokens", flush=True)
            
            if seq_len < max_N:
                print(f"WARNING: Prompt token length ({seq_len}) is smaller than the maximum requested sequence length ({max_N}). "
                      f"Metrics for N > {seq_len} will be skipped.", flush=True)
                run_len = seq_len
            else:
                if seq_len > max_N:
                    print(f"Trimming prompt token sequence from {seq_len} to max_N = {max_N} tokens before forward pass.", flush=True)
                run_len = max_N
            
            input_ids_sliced = input_ids[:, :run_len]
            attention_mask_sliced = attention_mask[:, :run_len]
            print(f"Running forward pass for length M = {run_len} tokens...", flush=True)
            
            # Dictionary to store captured arguments for each layer's self_attn
            layer_captures = {}
            original_forwards = {}
            
            # Function to intercept self_attn forward calls and capture arguments
            def make_custom_forward(idx, orig_f):
                def custom_forward(*args, **kwargs):
                    # Save to CPU to avoid GPU memory growth during forward pass
                    captured_args = []
                    for arg in args:
                        if isinstance(arg, torch.Tensor):
                            captured_args.append(arg.detach().clone().cpu())
                        else:
                            captured_args.append(arg)
                            
                    captured_kwargs = {}
                    for k, v in kwargs.items():
                        if isinstance(v, torch.Tensor):
                            captured_kwargs[k] = v.detach().clone().cpu()
                        elif isinstance(v, tuple):
                            captured_kwargs[k] = tuple(x.detach().clone().cpu() if isinstance(x, torch.Tensor) else x for x in v)
                        else:
                            captured_kwargs[k] = v
                            
                    layer_captures[idx] = (captured_args, captured_kwargs)
                    return orig_f(*args, **kwargs)
                return custom_forward
            
            num_layers = model.config.num_hidden_layers
            
            # Monkeypatch self_attn.forward to capture inputs
            for i in range(num_layers):
                attn_module = model.model.layers[i].self_attn
                original_forwards[i] = attn_module.forward
                attn_module.forward = make_custom_forward(i, original_forwards[i])
                
            with torch.no_grad():
                _ = model(input_ids=input_ids_sliced, attention_mask=attention_mask_sliced, use_cache=False)
                
            # Restore original forward methods
            for i in range(num_layers):
                model.model.layers[i].self_attn.forward = original_forwards[i]
                
            print("Computing Attention Jacobians and weight metrics layer-by-layer...", flush=True)
            
            results = {
                "layers": {}
            }
            
            dynamics_results = {}
            for n in N_list:
                dynamics_results[str(n)] = {}
            
            # Create a dedicated directory naming indicating the sequence lengths N
            run_name = f"{setup['name']}_N-{'-'.join(map(str, N_list))}_{prompt_hash}"
            setup_dir = os.path.join(base_results_dir, run_name)
            ensure_dir(setup_dir)
            
            output_file = os.path.join(setup_dir, f"attn_jacobian_measurements.json")
            dynamics_output_file = os.path.join(setup_dir, "attention_dynamics_results.json")
            
            for layer_idx in range(num_layers):
                captured_args, captured_kwargs = layer_captures[layer_idx]
                
                # Compute exact metrics for all valid n in N_list
                metrics = compute_attn_jacobian_metrics(
                    model, layer_idx, captured_args, captured_kwargs, N_list, K, device
                )
                
                # Format a print summary for the current layer
                summary_str = f"Layer {layer_idx:02d} | W_Q_K^T: {metrics['routing_weight_norm']:.4f} | W_V_O^T: {metrics['mixing_weight_norm']:.4f} | "
                len_summaries = []
                for n_val, n_met in metrics["seq_lengths"].items():
                    len_summaries.append(f"N={n_val} ||J||_2={n_met['attn_spectral_norm']:.3f}")
                summary_str += " | ".join(len_summaries)
                print(summary_str, flush=True)
                
                results["layers"][layer_idx] = metrics
                
                # Store in the dynamics results hierarchical structure
                for n in N_list:
                    if n in metrics["seq_lengths"]:
                        n_met = metrics["seq_lengths"][n]
                        dynamics_results[str(n)][str(layer_idx)] = {
                            "attn_spectral_norm": n_met["attn_spectral_norm"],
                            "mean_attn_entropy": n_met["mean_attn_entropy"],
                            "min_attn_entropy": n_met["min_attn_entropy"],
                            "max_attn_entropy": n_met["max_attn_entropy"],
                            "entropy_ratio": n_met["entropy_ratio"],
                            "mean_max_weight": n_met["mean_max_weight"],
                            "x_norm_mean": n_met["x_norm_mean"],
                            "mean_spectral_gap": n_met["mean_spectral_gap"],
                            "routing_weight_norm": metrics["routing_weight_norm"],
                            "mixing_weight_norm": metrics["mixing_weight_norm"],
                            "token_sensitivity_profile": n_met["token_sensitivity_profile"]
                        }
                
                # Incrementally save results after each layer
                with open(output_file, "w") as f:
                    json.dump(results, f, indent=4)
                
                with open(dynamics_output_file, "w") as f:
                    json.dump(dynamics_results, f, indent=4)
                
                # Clear memory
                del captured_args, captured_kwargs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            # Save metadata/config for this run
            metadata = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "setup": setup,
                "prompt_hash": prompt_hash,
                "prompt_text": prompt_text,
                "seq_len": seq_len,
                "N_list": N_list,
                "K": K
            }
            with open(os.path.join(setup_dir, "config.json"), "w") as f:
                json.dump(metadata, f, indent=4)
                
            print(f"Measurements saved successfully to {setup_dir}", flush=True)
            print(f"Dynamics results exported to JSON at: {dynamics_output_file}", flush=True)

if __name__ == "__main__":
    main()
