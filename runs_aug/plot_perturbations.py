import os
import glob
import json
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy import stats

def get_metric(distances_array, metric_type):
    """Calculates the specified metric for a flat array of distances."""
    if metric_type == "mean":
        return np.mean(distances_array)
    elif metric_type == "median":
        return np.median(distances_array)
    elif metric_type == "mode":
        rounded = np.round(distances_array, decimals=4)
        mode_result = stats.mode(rounded, keepdims=False)
        return float(mode_result.mode)
    else:
        raise ValueError(f"Unknown metric: {metric_type}")

def get_error(distances_array, error_type):
    """Calculates the specified error/variance metric."""
    if error_type == "std":
        return np.std(distances_array)
    elif error_type == "var":
        return np.var(distances_array)
    elif error_type == "none":
        return 0.0
    else:
        raise ValueError(f"Unknown error type: {error_type}")

def analyze_perturbations(base_results_dir, metric="mean", error_bars="none", plot_prompt_together=False):
    """
    Crawls the base_results_dir for all subfolders with config.json.
    Aggregates results by setup (and prompt, if requested) pooling irrelevant parameters like seeds.
    Plots lines for different perturbation radii.
    """
    if not os.path.exists(base_results_dir):
        print(f"Error: Directory {base_results_dir} does not exist.")
        return
        
    config_files = glob.glob(os.path.join(base_results_dir, "*", "config.json"))
    if not config_files:
        print(f"No config.json files found in subdirectories of {base_results_dir}.")
        return

    # Structure to hold full trajectories:
    # grouped_trajectories[group_key][radius] = list of tuples (layers, traj_array)
    # where traj_array is [num_layers, num_points]
    grouped_trajectories = defaultdict(lambda: defaultdict(list))
    group_titles = {}
    
    print(f"Found {len(config_files)} run configurations. Processing tensors...")
    
    for cfg_path in config_files:
        run_dir = os.path.dirname(cfg_path)
        with open(cfg_path, "r") as f:
            config = json.load(f)
            
        setup_name = config["setup"]["name"]
        prompt_idx = config.get("prompt_idx", 0)
        prompt_text = config.get("prompt_text", "Unknown Prompt")
        radius = config["radius"]
        
        # Grouping Abstraction
        # If plot_prompt_together is True, prompts become an irrelevant parameter and are pooled
        if plot_prompt_together:
            group_key = f"{setup_name}"
            title = f"Setup: {setup_name} (All Prompts Aggregated)"
        else:
            group_key = f"{setup_name}_p{prompt_idx}"
            short_prompt = prompt_text if len(prompt_text) < 40 else prompt_text[:37] + "..."
            title = f"Setup: {setup_name} | Prompt: '{short_prompt}'"
            
        group_titles[group_key] = title
        
        # Load all .pt files for this run
        layer_files = glob.glob(os.path.join(run_dir, "*.pt"))
        layer_dict = {}
        for lf in layer_files:
            basename = os.path.basename(lf)
            try:
                layer_idx = int(basename.split("_layer_")[-1].replace(".pt", ""))
                layer_dict[layer_idx] = lf
            except ValueError:
                continue
                
        sorted_layers = sorted(layer_dict.keys())
        if not sorted_layers:
            continue
            
        # Build [num_layers, num_points] distance array to preserve trajectory information
        run_layer_distances = []
        valid_layers = []
        for l_idx in sorted_layers:
            states = torch.load(layer_dict[l_idx], weights_only=True)
            base_state = states[0]
            perturbed_states = states[1:]
            
            if perturbed_states.shape[0] == 0:
                continue
                
            diffs = perturbed_states - base_state.unsqueeze(0)
            distances = torch.norm(diffs, p=2, dim=-1) # [n_conditions-1, seq_len]
            
            run_layer_distances.append(distances.flatten().cpu().numpy())
            valid_layers.append(l_idx)
            
        if run_layer_distances:
            traj_array = np.stack(run_layer_distances, axis=0)
            grouped_trajectories[group_key][radius].append((valid_layers, traj_array))
            
    plots_dir = os.path.join(base_results_dir, "aggregated_plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot Generation
    for group_key, radii_data in grouped_trajectories.items():
        plt.figure(figsize=(10, 6))
        sorted_radii = sorted(radii_data.keys())
        
        # Colormap abstraction
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(sorted_radii), 10)))
        
        for i, radius in enumerate(sorted_radii):
            color = colors[i % len(colors)]
            runs_list = radii_data[radius]
            
            # Aggregate points per layer across all irrelevant pooled parameters (seeds, prompts)
            layer_to_points = defaultdict(list)
            for (layers, traj_array) in runs_list:
                for l_idx, l_data in zip(layers, traj_array):
                    layer_to_points[l_idx].extend(l_data)
                    
            if not layer_to_points:
                continue
                
            sorted_layers = sorted(layer_to_points.keys())
            layer_arr = np.array(sorted_layers)
            
            if metric == "individual":
                # Plot every single trajectory as a faint line
                for (layers, traj_array) in runs_list:
                    plt.plot(layers, traj_array, color=color, alpha=0.08, linewidth=0.5)
                # Plot an invisible line just to get the legend to show
                plt.plot([], [], color=color, label=f"{radius}")
            else:
                # Plot aggregated metric
                m_vals = []
                e_vals = []
                for l_idx in sorted_layers:
                    pts = np.array(layer_to_points[l_idx])
                    m_vals.append(get_metric(pts, metric))
                    e_vals.append(get_error(pts, error_bars))
                    
                m_arr = np.array(m_vals)
                e_arr = np.array(e_vals)
                
                plt.plot(layer_arr, m_arr, marker='o', color=color, label=f"{radius}")
                if error_bars != "none":
                    plt.fill_between(layer_arr, m_arr - e_arr, m_arr + e_arr, color=color, alpha=0.2)
                    
        title = group_titles[group_key]
        plt.title(f"Divergence over Layers | {title}", fontsize=14)
        plt.xlabel("Layer Index", fontsize=12)
        
        ylabel = "L2 Distance"
        if metric != "individual":
            ylabel = f"{metric.capitalize()} " + ylabel
            if error_bars != "none":
                ylabel += f" (± {error_bars})"
                
        plt.ylabel(ylabel, fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(title="Magnitude")
        plt.tight_layout()
        
        # Sanitize filename
        safe_key = group_key.replace(" ", "_").replace("/", "-")
        plot_filename = f"{safe_key}_metric-{metric}_err-{error_bars}.png"
        plot_path = os.path.join(plots_dir, plot_filename)
        plt.savefig(plot_path, dpi=300)
        print(f"Generated plot: {plot_path}")
        plt.close()

def main():
    config_path = "config.yaml"
    if not os.path.exists(config_path):
        print(f"Error: {config_path} not found. Are you running from the right directory?")
        return
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    results_dir = config.get("experiment", {}).get("results_dir", "./results_perturbations")
    analysis_cfg = config.get("analysis", {})
    
    metric = analysis_cfg.get("metric", "mean")
    error_bars = analysis_cfg.get("error_bars", "none")
    plot_prompt_together = analysis_cfg.get("plot_prompt_together", False)
    
    if metric == "individual":
        error_bars = "none"
        
    print(f"Loaded config | Metric: {metric} | Error Bars: {error_bars} | Aggregating Prompts: {plot_prompt_together}")
    analyze_perturbations(results_dir, metric=metric, error_bars=error_bars, plot_prompt_together=plot_prompt_together)

if __name__ == "__main__":
    main()
