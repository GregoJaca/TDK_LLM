import os
import glob
import json
import yaml
import numpy as np
from collections import defaultdict
from scipy import stats
import pickle

def get_metrics_and_errors(data_array):
    """Calculates all metrics and errors for a flat array."""
    mean_val = np.mean(data_array)
    median_val = np.median(data_array)
    
    rounded = np.round(data_array, decimals=4)
    mode_result = stats.mode(rounded, keepdims=False)
    mode_val = float(mode_result.mode)
    
    std_val = np.std(data_array)
    var_val = np.var(data_array)
    
    return {
        "mean": float(mean_val),
        "median": float(median_val),
        "mode": float(mode_val),
        "std": float(std_val),
        "var": float(var_val),
        "none": 0.0
    }

def analyze_jacobians(base_results_dir):
    if not os.path.exists(base_results_dir):
        print(f"Error: Directory {base_results_dir} does not exist.")
        return
        
    config_files = glob.glob(os.path.join(base_results_dir, "*", "config.json"))
    if not config_files:
        print(f"No config.json files found in subdirectories of {base_results_dir}.")
        return

    # For grouping
    group_titles = {}
    
    # metrics_data[group_key][metric_name][layer_idx] = list of arrays
    metrics_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    # We also have scalar metrics that don't need aggregation across tokens, but might need it across prompts
    scalar_metrics_names = ["W_gate_max_SVD", "W_up_max_SVD", "W_down_max_SVD", 
                            "W_gate_scaled_F2", "W_up_scaled_F2", "W_down_scaled_F2"]
    token_metrics_names = ["spectral_norms", "lambda_true", "S_x_sq_mean", "D_x_sq_mean"]
    
    print(f"Found {len(config_files)} run configurations. Extracting Jacobian metrics...")
    
    for cfg_path in config_files:
        run_dir = os.path.dirname(cfg_path)
        with open(cfg_path, "r") as f:
            config = json.load(f)
            
        setup_name = config["setup"]["name"]
        prompt_hash = config.get("prompt_hash", "unknown")
        prompt_text = config.get("prompt_text", "Unknown Prompt")
        
        group_key_sep = f"{setup_name}_{prompt_hash}"
        short_prompt = prompt_text if len(prompt_text) < 40 else prompt_text[:37] + "..."
        group_titles[group_key_sep] = f"Setup: {setup_name} | Prompt: '{short_prompt}'"
        
        group_key_tog = f"{setup_name}_aggregated"
        group_titles[group_key_tog] = f"Setup: {setup_name} (All Prompts Aggregated)"
        
        json_path = os.path.join(run_dir, "mlp_jacobian_measurements.json")
        if not os.path.exists(json_path):
            continue
            
        with open(json_path, "r") as f:
            jdata = json.load(f)
            
        layers = sorted([int(k) for k in jdata["layers"].keys()])
        
        for l_idx in layers:
            layer_data = jdata["layers"][str(l_idx)]
            
            # Token metrics
            for m_name in token_metrics_names:
                if m_name in ["S_x_sq_mean", "D_x_sq_mean"]:
                    val = np.array(layer_data["activation_density"][m_name])
                else:
                    val = np.array(layer_data[m_name])
                metrics_data[group_key_sep][m_name][l_idx].append(val)
                metrics_data[group_key_tog][m_name][l_idx].append(val)
                
            # Scalar metrics
            for m_name in scalar_metrics_names:
                val = np.array([layer_data["weight_metrics"][m_name]]) # Wrap in array for consistency
                metrics_data[group_key_sep][m_name][l_idx].append(val)
                metrics_data[group_key_tog][m_name][l_idx].append(val)
                
    print("Computing metrics and aggregating data...")
    analyzed_data = {
        "group_titles": group_titles,
        "data": defaultdict(lambda: defaultdict(dict)),
        "scalar_metrics": scalar_metrics_names,
        "token_metrics": token_metrics_names
    }
    
    for group_key, m_dict in metrics_data.items():
        for m_name, layer_dict in m_dict.items():
            sorted_layers = sorted(layer_dict.keys())
            layer_arr = np.array(sorted_layers)
            
            metrics_per_layer = {
                "mean": [], "median": [], "mode": [], "std": [], "var": [], "none": []
            }
            
            for l_idx in sorted_layers:
                pts = np.concatenate(layer_dict[l_idx])
                metrics = get_metrics_and_errors(pts)
                for k, v in metrics.items():
                    metrics_per_layer[k].append(v)
                    
            metrics_per_layer = {k: np.array(v) for k, v in metrics_per_layer.items()}
            metrics_per_layer["layers"] = layer_arr
            
            analyzed_data["data"][group_key][m_name] = metrics_per_layer

    analyzed_data["data"] = {k: dict(v) for k, v in analyzed_data["data"].items()}
    
    out_file = os.path.join(base_results_dir, "analyzed_jacobians.pkl")
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
        
    results_dir = config.get("experiment", {}).get("results_dir", "./results_jacobians")
    
    print(f"Starting Jacobian analysis...")
    analyze_jacobians(results_dir)

if __name__ == "__main__":
    main()
