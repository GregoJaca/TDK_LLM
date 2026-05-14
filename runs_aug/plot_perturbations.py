import os
import glob
import json
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy import stats

def get_metric(distances, metric_type):
    """Calculates the specified metric for a flat array of distances."""
    if metric_type == "mean":
        return distances.mean().item()
    elif metric_type == "median":
        return distances.median().item()
    elif metric_type == "mode":
        # Mode of continuous floats doesn't make strict sense, but we can compute a kernel density estimate or histogram peak.
        # As a fast fallback, we bin the data and take the center of the largest bin, or use scipy stats mode.
        # Scipy mode on floats will just find exact duplicates, which is rare. Better to round.
        rounded = np.round(distances.cpu().numpy(), decimals=4)
        mode_result = stats.mode(rounded, keepdims=False)
        return float(mode_result.mode)
    else:
        raise ValueError(f"Unknown metric: {metric_type}")

def get_error(distances, error_type):
    """Calculates the specified error/variance metric."""
    if error_type == "std":
        return distances.std().item()
    elif error_type == "var":
        return distances.var().item()
    elif error_type == "none":
        return 0.0
    else:
        raise ValueError(f"Unknown error type: {error_type}")

def aggregate_and_plot(base_results_dir, metric="mean", error_bars="none"):
    """
    Crawls the base_results_dir for all subfolders with config.json.
    Aggregates results by setup and prompt.
    Plots lines for different perturbation radii.
    """
    if not os.path.exists(base_results_dir):
        print(f"Error: Directory {base_results_dir} does not exist.")
        return
        
    # Find all config.json files in subdirectories
    config_files = glob.glob(os.path.join(base_results_dir, "*", "config.json"))
    
    if not config_files:
        print(f"No config.json files found in subdirectories of {base_results_dir}.")
        return

    # Data structure to group runs:
    # grouped_data[setup_name][prompt_idx][radius][layer_idx] = (metric_val, error_val)
    grouped_data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    metadata_map = {} # Store metadata for plotting titles
    
    print(f"Found {len(config_files)} run configurations. Processing tensors...")
    
    for cfg_path in config_files:
        run_dir = os.path.dirname(cfg_path)
        with open(cfg_path, "r") as f:
            config = json.load(f)
            
        setup_name = config["setup"]["name"]
        prompt_idx = config.get("prompt_idx", 0)
        radius = config["radius"]
        
        # Save metadata for later
        group_key = (setup_name, prompt_idx)
        if group_key not in metadata_map:
            metadata_map[group_key] = config
            
        # Find all .pt files in this run_dir
        layer_files = glob.glob(os.path.join(run_dir, "*.pt"))
        for lf in layer_files:
            basename = os.path.basename(lf)
            try:
                layer_idx = int(basename.split("_layer_")[-1].replace(".pt", ""))
            except ValueError:
                continue
                
            # Load tensor and compute distances
            states = torch.load(lf, weights_only=True) # [n_conditions, seq_len, hidden_size]
            
            base_state = states[0]
            perturbed_states = states[1:]
            
            if perturbed_states.shape[0] == 0:
                continue
                
            # L2 Distance across hidden size
            diffs = perturbed_states - base_state.unsqueeze(0)
            distances = torch.norm(diffs, p=2, dim=-1) # [n_conditions-1, seq_len]
            
            # Flatten to compute metric across all perturbed copies and all tokens
            flat_distances = distances.flatten()
            
            m_val = get_metric(flat_distances, metric)
            e_val = get_error(flat_distances, error_bars)
            
            grouped_data[setup_name][prompt_idx][radius][layer_idx] = (m_val, e_val)

    # Ensure plots directory exists
    plots_dir = os.path.join(base_results_dir, "aggregated_plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Now generate one plot per setup_name + prompt_idx
    for setup_name, prompts_data in grouped_data.items():
        for prompt_idx, radii_data in prompts_data.items():
            
            meta = metadata_map[(setup_name, prompt_idx)]
            prompt_text = meta.get("prompt_text", "Unknown Prompt")
            
            plt.figure(figsize=(10, 6))
            
            # Sort radii so legend is ordered
            sorted_radii = sorted(radii_data.keys())
            
            # Colormap for different radii
            colors = plt.cm.tab10(np.linspace(0, 1, max(len(sorted_radii), 10)))
            
            for i, radius in enumerate(sorted_radii):
                layer_data = radii_data[radius]
                sorted_layers = sorted(layer_data.keys())
                
                if not sorted_layers:
                    continue
                    
                m_vals = [layer_data[l][0] for l in sorted_layers]
                e_vals = [layer_data[l][1] for l in sorted_layers]
                
                layer_arr = np.array(sorted_layers)
                m_arr = np.array(m_vals)
                e_arr = np.array(e_vals)
                
                color = colors[i % len(colors)]
                
                plt.plot(layer_arr, m_arr, marker='o', color=color, label=f"Radius = {radius}")
                
                if error_bars != "none":
                    plt.fill_between(layer_arr, m_arr - e_arr, m_arr + e_arr, color=color, alpha=0.2)
                    
            short_prompt = prompt_text if len(prompt_text) < 40 else prompt_text[:37] + "..."
            plt.title(f"Divergence over Layers | Setup: {setup_name}\nPrompt: '{short_prompt}'", fontsize=14)
            plt.xlabel("Layer Index", fontsize=12)
            ylabel = f"{metric.capitalize()} L2 Distance"
            if error_bars != "none":
                ylabel += f" (± {error_bars})"
            plt.ylabel(ylabel, fontsize=12)
            
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            
            plot_filename = f"{setup_name}_p{prompt_idx}_metric-{metric}_err-{error_bars}.png"
            plot_path = os.path.join(plots_dir, plot_filename)
            plt.savefig(plot_path, dpi=300)
            print(f"Generated aggregated plot: {plot_path}")
            plt.close()

def main():
    parser = argparse.ArgumentParser(description="Aggregated Data Analysis for Perturbation Experiments.")
    parser.add_argument("--results_dir", type=str, default="./results_perturbations",
                        help="Path to the base directory containing all run subfolders.")
    parser.add_argument("--metric", type=str, choices=["mean", "median", "mode"], default="mean",
                        help="Which metric to plot for the distances.")
    parser.add_argument("--error_bars", type=str, choices=["std", "var", "none"], default="none",
                        help="Type of error bars to plot around the metric.")
    
    args = parser.parse_args()
    aggregate_and_plot(args.results_dir, metric=args.metric, error_bars=args.error_bars)

if __name__ == "__main__":
    main()
