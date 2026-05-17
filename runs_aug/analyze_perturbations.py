import os
import glob
import json
import yaml
import torch
import numpy as np
from collections import defaultdict
from scipy import stats
import pickle

def get_metrics_and_errors(data_array):
    """Calculates all metrics and errors for a flat array of data."""
    if len(data_array) == 0:
        return {k: 0.0 for k in ["mean", "median", "mode", "std", "var", "min", "max", "p10", "p25", "p75", "p90", "none"]}

    mean_val = np.mean(data_array)
    median_val = np.median(data_array)
    
    # Mode is slow on large arrays, skip if too big or use a sample
    if len(data_array) > 10000:
        sample = np.random.choice(data_array, 10000, replace=False)
        rounded = np.round(sample, decimals=4)
    else:
        rounded = np.round(data_array, decimals=4)
    
    mode_result = stats.mode(rounded, keepdims=False)
    mode_val = float(mode_result.mode)
    
    std_val = np.std(data_array)
    var_val = np.var(data_array)
    min_val = np.min(data_array)
    max_val = np.max(data_array)
    
    p10, p25, p75, p90 = np.percentile(data_array, [10, 25, 75, 90])
    
    return {
        "mean": float(mean_val),
        "median": float(median_val),
        "mode": float(mode_val),
        "std": float(std_val),
        "var": float(var_val),
        "min": float(min_val),
        "max": float(max_val),
        "p10": float(p10),
        "p25": float(p25),
        "p75": float(p75),
        "p90": float(p90),
        "none": 0.0
    }

def analyze_perturbations(base_results_dir, distance_metric="L2", eval_tokens="last"):
    if not os.path.exists(base_results_dir):
        print(f"Error: Directory {base_results_dir} does not exist.")
        return
        
    config_files = glob.glob(os.path.join(base_results_dir, "*", "config.json"))
    if not config_files:
        print(f"No config.json files found in subdirectories of {base_results_dir}.")
        return

    # grouped_trajectories[group_key][radius] = list of tuples (layers, traj_array)
    grouped_trajectories = defaultdict(lambda: defaultdict(list))
    group_titles = {}
    
    print(f"Found {len(config_files)} run configurations. Calculating {distance_metric} distances...")
    
    for cfg_path in config_files:
        run_dir = os.path.dirname(cfg_path)
        with open(cfg_path, "r") as f:
            config = json.load(f)
            
        setup_name = config["setup"]["name"]
        prompt_hash = config.get("prompt_hash", "unknown")
        prompt_text = config.get("prompt_text", "Unknown Prompt")
        radius = config["radius"]
        
        group_key_sep = f"{setup_name}_{prompt_hash}"
        short_prompt = prompt_text if len(prompt_text) < 40 else prompt_text[:37] + "..."
        group_titles[group_key_sep] = f"Setup: {setup_name} | Prompt: '{short_prompt}'"
        
        group_key_tog = f"{setup_name}_aggregated"
        group_titles[group_key_tog] = f"Setup: {setup_name} (All Prompts Aggregated)"
        
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
            
        run_layer_distances = []
        valid_layers = []
        for l_idx in sorted_layers:
            states = torch.load(layer_dict[l_idx], weights_only=True)
            base_state = states[0]
            perturbed_states = states[1:]
            
            if perturbed_states.shape[0] == 0:
                continue
                
            if eval_tokens == "last":
                target_states = perturbed_states[:, -1, :].contiguous()
            else:
                target_states = perturbed_states.view(perturbed_states.shape[0], -1).contiguous()
                
            if distance_metric == "L2":
                target_distances = torch.nn.functional.pdist(target_states, p=2).cpu().numpy()
            elif distance_metric == "cos":
                target_states_norm = torch.nn.functional.normalize(target_states, p=2, dim=-1)
                cos_sim = torch.mm(target_states_norm, target_states_norm.t())
                indices = torch.triu_indices(cos_sim.size(0), cos_sim.size(1), offset=1)
                cos_distances = 1.0 - cos_sim[indices[0], indices[1]]
                target_distances = cos_distances.cpu().numpy()
            else:
                raise ValueError(f"Unknown distance_metric: {distance_metric}")
            
            run_layer_distances.append(target_distances)
            valid_layers.append(l_idx)
            
        if run_layer_distances:
            traj_array = np.stack(run_layer_distances, axis=0)
            grouped_trajectories[group_key_sep][radius].append((valid_layers, traj_array))
            grouped_trajectories[group_key_tog][radius].append((valid_layers, traj_array))
            
    print("Computing metrics and aggregating data...")
    # Now compute aggregated metrics per layer
    analyzed_data = {}
    analyzed_data["group_titles"] = group_titles
    analyzed_data["data"] = defaultdict(lambda: defaultdict(dict))
    
    for group_key, radii_data in grouped_trajectories.items():
        for radius, runs_list in radii_data.items():
            layer_to_arrays = defaultdict(list)
            for (layers, traj_array) in runs_list:
                for l_idx, l_data in zip(layers, traj_array):
                    layer_to_arrays[l_idx].append(l_data)
                    
            if not layer_to_arrays:
                continue
                
            sorted_layers = sorted(layer_to_arrays.keys())
            layer_arr = np.array(sorted_layers)
            
            metrics_per_layer = {
                "mean": [], "median": [], "mode": [], "std": [], "var": [], 
                "min": [], "max": [], "p10": [], "p25": [], "p75": [], "p90": [],
                "none": [], "hist": [], "hist_bins": []
            }
            
            for l_idx in sorted_layers:
                pts = np.concatenate(layer_to_arrays[l_idx])
                metrics = get_metrics_and_errors(pts)
                for k, v in metrics.items():
                    metrics_per_layer[k].append(v)
                
                # Compute histogram for distribution heatmap
                # Safely handle potential NaNs, infs, and tiny variance (peak-to-peak < 1e-7)
                pts_clean = pts[np.isfinite(pts)] if pts is not None else np.array([])
                if len(pts_clean) == 0:
                    pts_clean = np.array([0.0])
                
                pts_range = np.ptp(pts_clean)
                if pts_range < 1e-7:
                    val = pts_clean[0]
                    if val == 0.0:
                        b = np.linspace(-0.05, 0.05, 101)
                    else:
                        b = np.linspace(val * 0.95, val * 1.05, 101)
                    h = np.zeros(100)
                    h[50] = 1.0 / (b[51] - b[50])
                else:
                    h, b = np.histogram(pts_clean, bins=100, density=True)
                metrics_per_layer["hist"].append(h)
                metrics_per_layer["hist_bins"].append(b)
            
            # Convert to numpy arrays for storage
            metrics_per_layer = {k: np.array(v) for k, v in metrics_per_layer.items()}
            metrics_per_layer["layers"] = layer_arr
            
            analyzed_data["data"][group_key][radius] = metrics_per_layer

    # Convert defaultdict back to dict for clean saving
    analyzed_data["data"] = {k: dict(v) for k, v in analyzed_data["data"].items()}
    
    out_file = os.path.join(base_results_dir, "analyzed_data.pkl")
    with open(out_file, "wb") as f:
        pickle.dump(analyzed_data, f)
    print(f"Data analysis complete. Results saved to {out_file}")

def main():
    config_path = "config.yaml"
    if not os.path.exists(config_path):
        print(f"Error: {config_path} not found.")
        return
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    results_dir = config.get("experiment", {}).get("results_dir", "./results_perturbations")
    analysis_cfg = config.get("analysis", {})
    
    distance_metric = analysis_cfg.get("distance_metric", "L2")
    eval_tokens = analysis_cfg.get("eval_tokens", "last")
    
    print(f"Starting analysis | Metric: {distance_metric} | Eval: {eval_tokens}")
    analyze_perturbations(results_dir, distance_metric=distance_metric, eval_tokens=eval_tokens)

if __name__ == "__main__":
    main()
