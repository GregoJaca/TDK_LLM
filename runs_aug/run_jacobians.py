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
    # Load configuration
    config_path = "jacobian_config.yaml"
    if os.path.exists(config_path):
        config = load_config(config_path)
    else:
        # Fallback default configuration
        config = {
            "model": {"name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "local_dir": "."},
            "prompts": [
                {
                    "id": "chaos_theory_prompt",
                    "text": (
                        "Chaos theory is an interdisciplinary area of scientific study and branch of mathematics "
                        "focused on underlying patterns and deterministic laws of dynamical systems that are highly "
                        "sensitive to initial conditions, and were once thought to have completely random states of "
                        "disorder and irregularities. Chaos theory states that within the apparent randomness of "
                        "chaotic complex systems, there are underlying patterns, interconnectedness, constant "
                        "feedback loops, repetition, self-similarity, fractals, and self-organization. The "
                        "butterfly effect, an underlying principle of chaos, describes how a small change in one "
                        "state of a deterministic nonlinear system can result in large differences in a later state."
                    )
                }
            ]
        }
    
    model_name = config["model"].get("name", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    local_dir = config["model"].get("local_dir", ".")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results_dir = "./results_jacobians"
    ensure_dir(results_dir)
    
    print(f"Loading model: {model_name} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=local_dir)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16,
        cache_dir=local_dir
    )
    model.to(device)
    model.eval()
    
    for prompt_info in config.get("prompts", []):
        prompt_id = prompt_info["id"]
        prompt_text = prompt_info["text"]
        print(f"\n=== Processing Prompt: {prompt_id} ===")
        
        encoded = tokenizer(prompt_text, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        
        seq_len = input_ids.shape[1]
        print(f"Tokenized prompt length: {seq_len} tokens")
        
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
            
        print("Running forward pass to extract x_norm...")
        with torch.no_grad():
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
            
        for h in hooks:
            h.remove()
            
        print("Computing exact Jacobians and weight metrics layer-by-layer...")
        
        results = {
            "metadata": {
                "model": model_name,
                "prompt_id": prompt_id,
                "prompt": prompt_text,
                "seq_len": seq_len,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "layers": {}
        }
        
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
                  f"W_gate_F2/1536: {metrics['weight_metrics']['W_gate_scaled_F2']:.4f}")
                  
            results["layers"][layer_idx] = metrics
            
            # Clear memory immediately
            del x_norm
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        output_file = os.path.join(results_dir, f"mlp_jacobian_measurements_{prompt_id}.json")
        with open(output_file, "w") as f:
            json.dump(results, f, indent=4)
            
        print(f"Measurements saved successfully to {output_file}")

if __name__ == "__main__":
    main()
