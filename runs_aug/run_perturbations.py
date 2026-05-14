import os
import yaml
import json
import torch
import time
import hashlib
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.utils import PerformanceMonitor, ensure_dir
from src.perturbations import generate_simplex_perturbations
from src.generation_experiments import generate_and_save_hidden_states

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def run_experiment(config):
    monitor = PerformanceMonitor()
    monitor.start()

    model_config = config["model"]
    exp_config = config["experiment"]

    print(f"Loading model: {model_config['name']}")
    dtype = torch.float16 if model_config.get("dtype") == "float16" else torch.float32
    device = model_config.get("device", "cpu")

    tokenizer = AutoTokenizer.from_pretrained(
        model_config["name"], 
        cache_dir=model_config["local_dir"]
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_config["name"], 
        torch_dtype=dtype,
        cache_dir=model_config["local_dir"]
    )
    model.to(device)
    model.eval()

    num_layers = model.config.num_hidden_layers
    if exp_config["selected_layers"] == "all":
        selected_layers = list(range(num_layers + 1))
    else:
        selected_layers = exp_config["selected_layers"]

    base_results_dir = exp_config["results_dir"]
    ensure_dir(base_results_dir)

    prompts = {}
    for p in exp_config["prompts"]:
        if "texts" in p:
            prompts[p["id"]] = p["texts"]
        else:
            prompts[p["id"]] = [p["text"]]

    for setup in exp_config["setups"]:
        print(f"\n=== Running Setup: {setup['name']} ===")
        
        prompt_texts = prompts[setup["prompt_id"]]
        for prompt_idx, prompt_text in enumerate(prompt_texts):
            print(f"\n--- Prompt {prompt_idx + 1}/{len(prompt_texts)} ---")
            print(f"Text: '{prompt_text}'")
            
            prompt_hash = hashlib.md5(prompt_text.encode('utf-8')).hexdigest()[:8]
            
            encoded = tokenizer(prompt_text, return_tensors="pt")
            
            initial_tokens = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            
            if exp_config.get("remove_bos_token", False):
                # Pop the first token (usually the BOS token like <|begin_of_text|>)
                initial_tokens = initial_tokens[:, 1:]
                attention_mask = attention_mask[:, 1:]
            
            # Print tokenized strings for transparency
            token_ids = initial_tokens[0].tolist()
            token_strings = [tokenizer.decode([tid]) for tid in token_ids]
            print(f"Tokenized to ({len(token_strings)} tokens): {token_strings}")
            
            with torch.no_grad():
                base_embeds = model.get_input_embeddings()(initial_tokens)

            for radius in exp_config["radii"]:
                print(f"  Radius: {radius}")
                
                seeds = exp_config.get("seed", [42])
                if not isinstance(seeds, list):
                    seeds = [seeds]
                    
                for seed_val in seeds:
                    print(f"    Seed: {seed_val}")
                    
                    # Determine target tokens based on mode
                    if setup["mode"] == "all":
                        target_tokens = None
                    elif setup["mode"] == "first_m":
                        target_tokens = setup["m_tokens"]
                    elif setup["mode"] == "single_token":
                        target_tokens = 1
                    else:
                        raise ValueError(f"Unknown mode: {setup['mode']}")
                        
                    perturbed_embeddings = generate_simplex_perturbations(
                        base_embeds=base_embeds,
                        n_conditions=exp_config["n_conditions"],
                        radius=radius,
                        subspace_mode=exp_config["subspace_mode"],
                        target_tokens=target_tokens,
                        seed=seed_val
                    )
                    
                    # Create a dedicated directory for this specific configuration
                    run_name = f"{setup['name']}_{prompt_hash}_r{radius}_s{seed_val}"
                    setup_dir = os.path.join(base_results_dir, run_name)
                    ensure_dir(setup_dir)
                    
                    max_new_tokens = setup.get("max_new_tokens", 0)
                    batch_size = exp_config.get("batch_size", 64)
                    
                    generate_and_save_hidden_states(
                        model=model,
                        attention_mask=attention_mask,
                        perturbed_embeddings=perturbed_embeddings,
                        selected_layers=selected_layers,
                        results_dir=setup_dir,
                        run_name=run_name,
                        max_new_tokens=max_new_tokens,
                        batch_size=batch_size
                    )
                    
                    # Save metadata/config for this specific run inside its folder
                    metadata = {
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "setup": setup,
                        "prompt_hash": prompt_hash,
                        "radius": radius,
                        "n_conditions": exp_config["n_conditions"],
                        "subspace_mode": exp_config["subspace_mode"],
                        "prompt_text": prompt_text,
                        "tokenized_strings": token_strings,
                        "target_tokens_perturbed": target_tokens if target_tokens is not None else base_embeds.shape[1],
                        "max_new_tokens": max_new_tokens,
                        "seed": seed_val
                    }
                    with open(os.path.join(setup_dir, "config.json"), "w") as f:
                        json.dump(metadata, f, indent=4)

    monitor.end()

if __name__ == "__main__":
    config_path = "config.yaml"
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
    else:
        config = load_config(config_path)
        run_experiment(config)
