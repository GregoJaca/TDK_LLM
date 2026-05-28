import os
import yaml
import pickle
import numpy as np
from collections import defaultdict

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    config_path = "jacobian_config.yaml"
    if not os.path.exists(config_path):
        print(f"Error: {config_path} not found.")
        return
        
    config = load_config(config_path)
    norm_config = config.get("normalization", {})
    results_dir = norm_config.get("results_dir", "./results_normalization")
    
    raw_data_path = os.path.join(results_dir, "raw_normalization_data.pkl")
    if not os.path.exists(raw_data_path):
        print(f"Error: Raw data file {raw_data_path} not found. Run run_normalization_jacobians.py first.")
        return
        
    print(f"Loading raw data from {raw_data_path}...")
    with open(raw_data_path, "rb") as f:
        raw_data = pickle.load(f)
        
    data_list = raw_data["data"]
    radii = raw_data["radii"]
    num_layers = raw_data["num_layers"]
    
    # 1. Group data for aggregation
    # Keys for grouping: (norm_name, pert_type, radius) -> list of data dicts
    grouped = defaultdict(list)
    # Keys for layer-wise grouping: (norm_name, pert_type, layer) -> list of data dicts (usually at a small radius)
    layer_grouped = defaultdict(list)
    
    for item in data_list:
        norm_name = item["norm_name"]
        pert_type = item["pert_type"]
        radius = item["radius"]
        layer = item["layer"]
        
        grouped[(norm_name, pert_type, radius)].append(item)
        layer_grouped[(norm_name, pert_type, layer)].append(item)
        
    print("Aggregating metrics...")
    
    aggregated_sweep = {}
    
    # 2. Analyze magnitude sweep (log-log power laws)
    # We want to aggregate metrics for each (norm_name, pert_type, radius)
    for (norm_name, pert_type, radius), items in grouped.items():
        emp_norms = [i["emp_norm"] for i in items]
        jvp_norms = [i["jvp_norm"] for i in items]
        theory_norms = [i["theory_weighted_norm"] for i in items]
        
        cos_sims_emp_x = [i["cos_sim_emp_x"] for i in items]
        cos_sims_jvp_x = [i["cos_sim_jvp_x"] for i in items]
        cos_sims_emp_jvp = [i["cos_sim_emp_jvp"] for i in items]
        
        rel_errors_jvp = []
        for i in items:
            if i["jvp_norm"] > 0:
                rel_errors_jvp.append(abs(i["emp_norm"] - i["jvp_norm"]) / i["jvp_norm"])
            else:
                rel_errors_jvp.append(0.0)
                
        key = (norm_name, pert_type, radius)
        
        # Helper to compute statistics
        def get_stats(arr):
            arr = np.array(arr)
            return {
                "mean": float(np.mean(arr)),
                "median": float(np.median(arr)),
                "std": float(np.std(arr)),
                "p10": float(np.percentile(arr, 10)),
                "p25": float(np.percentile(arr, 25)),
                "p75": float(np.percentile(arr, 75)),
                "p90": float(np.percentile(arr, 90)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr))
            }
            
        aggregated_sweep[key] = {
            "norm_name": norm_name,
            "pert_type": pert_type,
            "radius": radius,
            "emp_norm": get_stats(emp_norms),
            "jvp_norm": get_stats(jvp_norms),
            "theory_norm": get_stats(theory_norms),
            "cos_sim_emp_x": get_stats(cos_sims_emp_x),
            "cos_sim_jvp_x": get_stats(cos_sims_jvp_x),
            "cos_sim_emp_jvp": get_stats(cos_sims_emp_jvp),
            "rel_err_jvp": get_stats(rel_errors_jvp)
        }
        
    # 3. Compute Power-Law Exponents and Linearity Boundaries
    # For each combination of norm_name and pert_type:
    power_laws = {}
    linearity_boundaries = {}
    
    unique_combinations = set((i["norm_name"], i["pert_type"]) for i in data_list)
    
    for norm_name, pert_type in unique_combinations:
        sweep_radii = []
        sweep_emp_medians = []
        sweep_rel_err_medians = []
        
        for radius in sorted(radii):
            key = (norm_name, pert_type, radius)
            if key in aggregated_sweep:
                sweep_radii.append(radius)
                sweep_emp_medians.append(aggregated_sweep[key]["emp_norm"]["median"])
                sweep_rel_err_medians.append(aggregated_sweep[key]["rel_err_jvp"]["median"])
                
        # Fit power law exponent in the linear regime (exclude very large/small radii)
        # We select radii in range [1e-7, 1e-2] for regression
        fit_x = []
        fit_y = []
        for r, val in zip(sweep_radii, sweep_emp_medians):
            if 1e-7 <= r <= 1e-2 and val > 0:
                fit_x.append(np.log10(r))
                fit_y.append(np.log10(val))
                
        exponent = None
        intercept = None
        if len(fit_x) >= 2:
            slope, intercept = np.polyfit(fit_x, fit_y, 1)
            exponent = float(slope)
            
        # Find the linearity boundary (radius at which relative error between empirical & JVP exceeds 5%)
        boundary_radius = None
        for r, err in zip(sweep_radii, sweep_rel_err_medians):
            # For radial perturbations, JVP norm is 0, so relative error might be undefined/noisy.
            # We look at the actual magnitude of empirical response relative to radius.
            if pert_type == "radial":
                # For radial, output should be close to 0. If it exceeds 1e-2 of radius, it's non-linear
                if sweep_emp_medians[sweep_radii.index(r)] / r > 1e-2:
                    boundary_radius = float(r)
                    break
            else:
                if err > 0.05:
                    boundary_radius = float(r)
                    break
                    
        combo_key = f"{norm_name}_{pert_type}"
        power_laws[combo_key] = {
            "exponent": exponent,
            "intercept": intercept
        }
        linearity_boundaries[combo_key] = boundary_radius
        
    # 4. Layer-wise validation metrics (for a chosen small radius to ensure linearity, e.g. 1e-4)
    # Target radius for scaling checks
    target_radius = 1e-4
    if target_radius not in radii:
        # fallback to the closest available radius
        idx = np.argmin(np.abs(np.array(radii) - target_radius))
        target_radius = radii[idx]
        
    print(f"Layer-wise analysis targeting radius = {target_radius}")
    
    layer_metrics = defaultdict(dict)
    
    for (norm_name, pert_type, layer), items in layer_grouped.items():
        # Filter for the target radius
        filtered = [i for i in items if i["radius"] == target_radius]
        if not filtered:
            continue
            
        emp_norms = [i["emp_norm"] for i in filtered]
        jvp_norms = [i["jvp_norm"] for i in filtered]
        theory_norms = [i["theory_weighted_norm"] for i in filtered]
        
        # Scaling ratios
        # Theoretical scaling factor (1 / S):
        # For RMS: sqrt(D) / ||x||_2
        # For LN: 1 / sigma
        # The stored theory_norm is computed as (1/S) * ||dx_perp||
        # Since dx_perp has norm target_radius (by design of orthogonal pert),
        # the theoretical ratio is theory_norm / target_radius.
        # Let's compute actual ratios:
        ratios_jvp = [i["jvp_norm"] / target_radius for i in filtered]
        ratios_emp = [i["emp_norm"] / target_radius for i in filtered]
        ratios_theory = [i["theory_weighted_norm"] / target_radius for i in filtered]
        
        cos_sims_emp_x = [i["cos_sim_emp_x"] for i in filtered]
        cos_sims_jvp_x = [i["cos_sim_jvp_x"] for i in filtered]
        
        def get_stats(arr):
            arr = np.array(arr)
            return {
                "mean": float(np.mean(arr)),
                "median": float(np.median(arr)),
                "std": float(np.std(arr)),
                "p10": float(np.percentile(arr, 10)),
                "p90": float(np.percentile(arr, 90))
            }
            
        layer_metrics[f"{norm_name}_{pert_type}"][layer] = {
            "jvp_ratio": get_stats(ratios_jvp),
            "emp_ratio": get_stats(ratios_emp),
            "theory_ratio": get_stats(ratios_theory),
            "cos_sim_emp_x": get_stats(cos_sims_emp_x),
            "cos_sim_jvp_x": get_stats(cos_sims_jvp_x)
        }
        
    analysis_results = {
        "model_name": raw_data["model_name"],
        "prompt_hash": raw_data["prompt_hash"],
        "num_layers": num_layers,
        "radii": radii,
        "target_radius": target_radius,
        "aggregated_sweep": aggregated_sweep,
        "power_laws": power_laws,
        "linearity_boundaries": linearity_boundaries,
        "layer_metrics": dict(layer_metrics)
    }
    
    analyzed_path = os.path.join(results_dir, "analyzed_normalization_data.pkl")
    with open(analyzed_path, "wb") as f:
        pickle.dump(analysis_results, f)
        
    print(f"Analysis complete. Results saved to {analyzed_path}")
    
    # Print printout summary for user
    print("\n--- Summary of Power Law Exponents ---")
    for combo_key, pl in power_laws.items():
        print(f"  {combo_key}: Exponent = {pl['exponent']:.4f} (expected: ~1.0000)")
        
    print("\n--- Summary of Linearity Boundaries ---")
    for combo_key, boundary in linearity_boundaries.items():
        print(f"  {combo_key}: Boundary = {boundary if boundary is not None else 'No boundary found (> max radius)'}")

if __name__ == "__main__":
    main()
