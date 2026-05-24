import os
import json
import yaml
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.jacobian_measurements import compute_mlp_jacobian_metrics

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
    
    model_name = model_config.get("name", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    local_dir = model_config.get("local_dir", ".")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_results_dir = exp_config.get("results_dir", "./results_jacobians")
    ensure_dir(base_results_dir)
    
    print(f"Loading model: {model_name} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=local_dir)
    
    # Retrieve dtype configuration
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
    
    prompts = {}
    for p in exp_config.get("prompts", []):
        if "texts" in p:
            prompts[p["id"]] = p["texts"]
        else:
            prompts[p["id"]] = [p["text"]]

    for setup in exp_config.get("setups", []):
        print(f"\n=== Running Setup: {setup['name']} ===", flush=True)
        
        prompt_texts = prompts[setup["prompt_id"]]
        for prompt_idx, prompt_text in enumerate(prompt_texts):
            import hashlib
            prompt_hash = hashlib.md5(prompt_text.encode('utf-8')).hexdigest()[:8]
            
            print(f"\n--- Prompt {prompt_idx + 1}/{len(prompt_texts)} ---", flush=True)
            
            encoded = tokenizer(prompt_text, return_tensors="pt")
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            
            seq_len = input_ids.shape[1]
            print(f"Tokenized prompt length: {seq_len} tokens", flush=True)
            
            # Hook into the post_attention_layernorm to capture x_norm
            layer_x_norms = {}
            
            def get_hook(layer_idx):
                def hook(module, input, output):
                    # Move to CPU immediately to free up GPU memory for later processing
                    layer_x_norms[layer_idx] = output.detach().clone().cpu()
                return hook
                
            hooks = []
            num_layers = model.config.num_hidden_layers
            for i in range(num_layers):
                h = model.model.layers[i].post_attention_layernorm.register_forward_hook(get_hook(i))
                hooks.append(h)
                
            print("Running forward pass to extract x_norm...", flush=True)
            with torch.no_grad():
                _ = model(input_ids=input_ids, attention_mask=attention_mask)
                
            for h in hooks:
                h.remove()
                
            print("Computing exact Jacobians and weight metrics layer-by-layer...", flush=True)
            
            results = {
                "layers": {}
            }
            
            # Create a dedicated directory early so we can save incrementally
            run_name = f"{setup['name']}_{prompt_hash}"
            setup_dir = os.path.join(base_results_dir, run_name)
            ensure_dir(setup_dir)
            
            output_file = os.path.join(setup_dir, f"mlp_jacobian_measurements.json")
            
            for layer_idx in range(num_layers):
                # Move only the current layer's x_norm to GPU
                x_norm = layer_x_norms[layer_idx].to(device)
                
                # Compute exact jacobian metrics
                metrics = compute_mlp_jacobian_metrics(model, x_norm, layer_idx)
                
                avg_spectral = sum(metrics["spectral_norms"]) / len(metrics["spectral_norms"])
                avg_lambda = sum(metrics["lambda_true"]) / len(metrics["lambda_true"])
                
                print(f"Layer {layer_idx:02d} | "
                      f"Avg ||J||_2: {avg_spectral:.4f} | "
                      f"Avg \u03bb_true: {avg_lambda:.4f} | "
                      f"W_gate_F2: {metrics['weight_metrics']['W_gate_scaled_F2']:.4f}", flush=True)
                      
                results["layers"][layer_idx] = metrics
                
                # Incrementally save results after each layer
                with open(output_file, "w") as f:
                    json.dump(results, f, indent=4)
                
                # Clear memory immediately
                del x_norm
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            # Save metadata/config for this specific run inside its folder
            metadata = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "setup": setup,
                "prompt_hash": prompt_hash,
                "prompt_text": prompt_text,
                "seq_len": seq_len
            }
            with open(os.path.join(setup_dir, "config.json"), "w") as f:
                json.dump(metadata, f, indent=4)
                
            print(f"Measurements saved successfully to {setup_dir}", flush=True)

if __name__ == "__main__":
    main()
