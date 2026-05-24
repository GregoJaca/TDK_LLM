import os
import yaml
import pickle
import numpy as np
import pandas as pd

def main():
    config_path = "jacobian_config.yaml"
    if not os.path.exists(config_path):
        print(f"Error: {config_path} not found.")
        return
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    results_dir = config.get("experiment", {}).get("results_dir", "./results_jacobians")
    data_path = os.path.join(results_dir, "analyzed_jacobians.pkl")
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return
        
    with open(data_path, "rb") as f:
        analyzed_data = pickle.load(f)
        
    data = analyzed_data["data"]
    
    for group_key, group_data in data.items():
        # Only process aggregated setups
        if not group_key.endswith("_aggregated"):
            continue
            
        metrics = {
            "W_gate_max_SVD": "W_gate_SVD",
            "W_up_max_SVD": "W_up_SVD",
            "W_down_max_SVD": "W_down_SVD",
            "W_gate_scaled_F2": "W_gate_Frobenius",
            "W_up_scaled_F2": "W_up_Frobenius",
            "W_down_scaled_F2": "W_down_Frobenius"
        }
        
        results = {}
        for metric_name, pretty_name in metrics.items():
            if metric_name in group_data:
                arr = group_data[metric_name]["mean"]
                results[pretty_name] = float(np.mean(arr))
                
        if not results:
            continue
            
        df = pd.DataFrame([
            {"Matrix": "W_gate", "SVD": results.get("W_gate_SVD"), "Frobenius": results.get("W_gate_Frobenius")},
            {"Matrix": "W_up", "SVD": results.get("W_up_SVD"), "Frobenius": results.get("W_up_Frobenius")},
            {"Matrix": "W_down", "SVD": results.get("W_down_SVD"), "Frobenius": results.get("W_down_Frobenius")}
        ])
        
        out_csv = os.path.join(results_dir, "weight_averages.csv")
        df.to_csv(out_csv, index=False)
        print(f"Saved weight averages to {out_csv}")
        print(df.to_string(index=False))

if __name__ == "__main__":
    main()
