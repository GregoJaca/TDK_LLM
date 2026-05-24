import os
import glob
import json
import yaml
import numpy as np
from collections import defaultdict
from scipy import stats
import pickle

def get_metrics_and_errors(data_array, axis=0):
    """Calculates all metrics and errors for an array of data along the specified axis."""
    data_array = np.array(data_array)
    if len(data_array) == 0:
        return {k: 0.0 for k in ["mean", "median", "mode", "harmonic", "std", "var", "harmonic_std", "harmonic_var", "min", "max", "p10", "p25", "p75", "p90", "none"]}

    mean_val = np.mean(data_array, axis=axis)
    median_val = np.median(data_array, axis=axis)
    std_val = np.std(data_array, axis=axis)
    var_val = np.var(data_array, axis=axis)
    min_val = np.min(data_array, axis=axis)
    max_val = np.max(data_array, axis=axis)
    
    pcts = np.percentile(data_array, [10, 25, 75, 90], axis=axis)
    p10, p25, p75, p90 = pcts[0], pcts[1], pcts[2], pcts[3]
    
    # Defaults for mode and harmonic metrics which only make sense for 1D scalars
    mode_val = 0.0
    harmonic_val = 0.0
    harmonic_std = 0.0
    harmonic_var = 0.0
    
    if data_array.ndim == 1:
        # Mode
        if len(data_array) > 10000:
            sample = np.random.choice(data_array, 10000, replace=False)
            rounded = np.round(sample, decimals=4)
        else:
            rounded = np.round(data_array, decimals=4)
        
        mode_result = stats.mode(rounded, keepdims=False)
        mode_val = float(mode_result.mode)
        
        # Harmonic mean
        data_positive = data_array[data_array > 0]
        if len(data_positive) > 0:
            n_h = len(data_positive)
            with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
                inv_data = 1.0 / data_positive
                sum_inv = np.sum(inv_data)
                if sum_inv > 0 and np.isfinite(sum_inv):
                    harmonic_val = n_h / sum_inv
                    var_x = np.var(data_positive)
                    inv_data_4 = inv_data ** 4
                    sum_inv4 = np.sum(inv_data_4)
                    if np.isfinite(sum_inv4):
                        harmonic_var = (harmonic_val ** 4 / (n_h ** 2)) * var_x * sum_inv4
                        if np.isfinite(harmonic_var) and harmonic_var >= 0:
                            harmonic_std = np.sqrt(harmonic_var)
                        else:
                            harmonic_var = 0.0
                            harmonic_std = 0.0
                    else:
                        harmonic_var = 0.0
                        harmonic_std = 0.0
                else:
                    harmonic_val = 0.0
                    harmonic_std = 0.0
                    harmonic_var = 0.0
        else:
            harmonic_val = 0.0
            harmonic_std = 0.0
            harmonic_var = 0.0
                
    return {
        "mean": mean_val,
        "median": median_val,
        "mode": mode_val,
        "harmonic": harmonic_val,
        "std": std_val,
        "var": var_val,
        "harmonic_std": harmonic_std,
        "harmonic_var": harmonic_var,
        "min": min_val,
        "max": max_val,
        "p10": p10,
        "p25": p25,
        "p75": p75,
        "p90": p90,
        "none": 0.0
    }

def analyze_jacobians_attn(base_results_dir):
    if not os.path.exists(base_results_dir):
        print(f"Error: Directory {base_results_dir} does not exist.")
        return
        
    config_files = glob.glob(os.path.join(base_results_dir, "*", "config.json"))
    if not config_files:
        print(f"No config.json files found in subdirectories of {base_results_dir}.")
        return
        
    group_titles = {}
    metrics_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    # We will dynamically find sequence lengths N by reading the data files
    found_N_sets = set()
    
    print(f"Found {len(config_files)} run configurations. Extracting Attention metrics...")
    
    for cfg_path in config_files:
        run_dir = os.path.dirname(cfg_path)
        with open(cfg_path, "r") as f:
            config = json.load(f)
            
        setup_name = config["setup"]["name"]
        prompt_hash = config.get("prompt_hash", "unknown")
        prompt_text = config.get("prompt_text", "Unknown Prompt")
        
        # Recover N_list from config or directory name
        N_list = config.get("N_list", [20, 100, 1000])
        found_N_sets.update(N_list)
        
        # Unique identifier for the list of N to keep different parameter runs separate
        n_str = "-".join(map(str, N_list))
        
        group_key_sep = f"{setup_name}_N-{n_str}_{prompt_hash}"
        short_prompt = prompt_text if len(prompt_text) < 40 else prompt_text[:37] + "..."
        group_titles[group_key_sep] = f"Setup: {setup_name} | N: {N_list} | Prompt: '{short_prompt}'"
        
        group_key_tog = f"{setup_name}_N-{n_str}_aggregated"
        group_titles[group_key_tog] = f"Setup: {setup_name} | N: {N_list} (All Prompts Aggregated)"
        
        json_path = os.path.join(run_dir, "attn_jacobian_measurements.json")
        if not os.path.exists(json_path):
            continue
            
        with open(json_path, "r") as f:
            jdata = json.load(f)
            
        layers = sorted([int(k) for k in jdata["layers"].keys()])
        
        for l_idx in layers:
            layer_data = jdata["layers"][str(l_idx)]
            
            # Static Weight Amplifiers (independent of N)
            metrics_data[group_key_sep]["routing_weight_norm"][l_idx].append(layer_data["routing_weight_norm"])
            metrics_data[group_key_tog]["routing_weight_norm"][l_idx].append(layer_data["routing_weight_norm"])
            
            metrics_data[group_key_sep]["mixing_weight_norm"][l_idx].append(layer_data["mixing_weight_norm"])
            metrics_data[group_key_tog]["mixing_weight_norm"][l_idx].append(layer_data["mixing_weight_norm"])
            
            # Dynamic metrics for each N
            for n_val in layer_data["seq_lengths"].keys():
                n_data = layer_data["seq_lengths"][n_val]
                n_int = int(n_val)
                
                metrics_data[group_key_sep][f"attn_spectral_norm_N-{n_int}"][l_idx].append(n_data["attn_spectral_norm"])
                metrics_data[group_key_tog][f"attn_spectral_norm_N-{n_int}"][l_idx].append(n_data["attn_spectral_norm"])
                
                metrics_data[group_key_sep][f"mean_attn_entropy_N-{n_int}"][l_idx].append(n_data["mean_attn_entropy"])
                metrics_data[group_key_tog][f"mean_attn_entropy_N-{n_int}"][l_idx].append(n_data["mean_attn_entropy"])
                
                metrics_data[group_key_sep][f"mean_spectral_gap_N-{n_int}"][l_idx].append(n_data["mean_spectral_gap"])
                metrics_data[group_key_tog][f"mean_spectral_gap_N-{n_int}"][l_idx].append(n_data["mean_spectral_gap"])
                
                metrics_data[group_key_sep][f"token_sensitivity_profile_N-{n_int}"][l_idx].append(n_data["token_sensitivity_profile"])
                metrics_data[group_key_tog][f"token_sensitivity_profile_N-{n_int}"][l_idx].append(n_data["token_sensitivity_profile"])
                
    print("Computing metrics and aggregating data...")
    
    analyzed_data = {
        "group_titles": group_titles,
        "data": defaultdict(lambda: defaultdict(dict)),
        "found_N_list": sorted(list(found_N_sets))
    }
    
    for group_key, m_dict in metrics_data.items():
        for m_name, layer_dict in m_dict.items():
            sorted_layers = sorted(layer_dict.keys())
            layer_arr = np.array(sorted_layers)
            
            # Create holders for the lists
            metrics_per_layer = defaultdict(list)
            
            for l_idx in sorted_layers:
                pts = layer_dict[l_idx]
                
                # Check if it's token_sensitivity_profile (which is 2D list of vectors)
                if "token_sensitivity_profile" in m_name:
                    # shape: (P, n) where P is the number of prompts in this group
                    pts_arr = np.array(pts)
                    metrics = get_metrics_and_errors(pts_arr, axis=0)
                    for k, v in metrics.items():
                        metrics_per_layer[k].append(v)
                    metrics_per_layer["raw"].append(pts_arr)
                else:
                    # shape: (P,)
                    pts_arr = np.array(pts)
                    metrics = get_metrics_and_errors(pts_arr, axis=0)
                    for k, v in metrics.items():
                        metrics_per_layer[k].append(v)
                    metrics_per_layer["raw"].append(pts_arr)
                    
            # Convert lists of arrays to final numpy arrays
            metrics_per_layer_converted = {}
            for k, v in metrics_per_layer.items():
                if k == "raw":
                    metrics_per_layer_converted[k] = v
                else:
                    metrics_per_layer_converted[k] = np.array(v)
            metrics_per_layer = metrics_per_layer_converted
            metrics_per_layer["layers"] = layer_arr
            
            analyzed_data["data"][group_key][m_name] = metrics_per_layer

    analyzed_data["data"] = {k: dict(v) for k, v in analyzed_data["data"].items()}
    
    out_file = os.path.join(base_results_dir, "analyzed_jacobians_attn.pkl")
    with open(out_file, "wb") as f:
        pickle.dump(analyzed_data, f)
    print(f"Data analysis complete. Results saved to {out_file}")

def main():
    config_path = "jacobian_config.yaml"
    if not os.path.exists(config_path):
        print(f"Error: {config_path} not found.")
        return
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    results_dir = config.get("experiment", {}).get("results_dir", "./results_jacobians_microsoft")
    
    print(f"Starting Attention Jacobian analysis...")
    analyze_jacobians_attn(results_dir)

if __name__ == "__main__":
    main()
