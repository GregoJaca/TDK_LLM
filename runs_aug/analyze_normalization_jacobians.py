import os
import yaml
import pickle
import numpy as np
from collections import defaultdict
from scipy import stats

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def get_metrics_and_errors(data_array):
    """Calculates all metrics and errors for a flat array of data."""
    data_array = np.array(data_array)
    if len(data_array) == 0:
        return {k: 0.0 for k in ["mean", "median", "mode", "harmonic", "std", "var", "harmonic_std", "harmonic_var", "min", "max", "p10", "p25", "p50", "p75", "p90", "none"]}

    mean_val = np.mean(data_array)
    median_val = np.median(data_array)
    
    # Mode
    if len(data_array) > 10000:
        sample = np.random.choice(data_array, 10000, replace=False)
        rounded = np.round(sample, decimals=4)
    else:
        rounded = np.round(data_array, decimals=4)
    
    mode_result = stats.mode(rounded, keepdims=False)
    mode_val = float(mode_result.mode)
    
    # Harmonic mean and propagated error
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

    std_val = np.std(data_array)
    var_val = np.var(data_array)
    min_val = np.min(data_array)
    max_val = np.max(data_array)
    
    p10, p25, p50, p75, p90 = np.percentile(data_array, [10, 25, 50, 75, 90])
    
    return {
        "mean": float(mean_val),
        "median": float(median_val),
        "p50": float(p50),
        "mode": float(mode_val),
        "harmonic": float(harmonic_val),
        "std": float(std_val),
        "var": float(var_val),
        "harmonic_std": float(harmonic_std),
        "harmonic_var": float(harmonic_var),
        "min": float(min_val),
        "max": float(max_val),
        "p10": float(p10),
        "p25": float(p25),
        "p75": float(p75),
        "p90": float(p90),
        "none": 0.0
    }

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
    grouped = defaultdict(list)
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
    
    # 2. Analyze magnitude sweep
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
        
        aggregated_sweep[key] = {
            "norm_name": norm_name,
            "pert_type": pert_type,
            "radius": radius,
            "emp_norm": get_metrics_and_errors(emp_norms),
            "jvp_norm": get_metrics_and_errors(jvp_norms),
            "theory_norm": get_metrics_and_errors(theory_norms),
            "cos_sim_emp_x": get_metrics_and_errors(cos_sims_emp_x),
            "cos_sim_jvp_x": get_metrics_and_errors(cos_sims_jvp_x),
            "cos_sim_emp_jvp": get_metrics_and_errors(cos_sims_emp_jvp),
            "rel_err_jvp": get_metrics_and_errors(rel_errors_jvp)
        }
        
    # 3. Compute Power-Law Exponents and Linearity Boundaries
    power_laws = {}
    linearity_boundaries = {}
    
    unique_combinations = set((i["norm_name"], i["pert_type"]) for i in data_list)
    
    for norm_name, pert_type in unique_combinations:
        for metric_name in ["mean", "median", "harmonic"]:
            sweep_radii = []
            sweep_emp_vals = []
            sweep_rel_err_vals = []
            
            for radius in sorted(radii):
                key = (norm_name, pert_type, radius)
                if key in aggregated_sweep:
                    sweep_radii.append(radius)
                    sweep_emp_vals.append(aggregated_sweep[key]["emp_norm"][metric_name])
                    sweep_rel_err_vals.append(aggregated_sweep[key]["rel_err_jvp"][metric_name])
                    
            fit_x = []
            fit_y = []
            for r, val in zip(sweep_radii, sweep_emp_vals):
                if 1e-7 <= r <= 1e-2 and val > 0:
                    fit_x.append(np.log10(r))
                    fit_y.append(np.log10(val))
                    
            # Fallback if too few points in range
            if len(fit_x) < 2:
                fit_x = []
                fit_y = []
                for r, val in zip(sweep_radii, sweep_emp_vals):
                    if val > 0:
                        fit_x.append(np.log10(r))
                        fit_y.append(np.log10(val))
                    
            exponent = None
            intercept = None
            if len(fit_x) >= 2:
                slope, intercept = np.polyfit(fit_x, fit_y, 1)
                exponent = float(slope)
                
            boundary_radius = None
            for r, err in zip(sweep_radii, sweep_rel_err_vals):
                if pert_type == "radial":
                    if sweep_emp_vals[sweep_radii.index(r)] / r > 1e-2:
                        boundary_radius = float(r)
                        break
                else:
                    if err > 0.05:
                        boundary_radius = float(r)
                        break
                        
            combo_key = f"{norm_name}_{pert_type}_{metric_name}"
            power_laws[combo_key] = {
                "exponent": exponent,
                "intercept": intercept
            }
            linearity_boundaries[combo_key] = boundary_radius
        
    # 4. Layer-wise validation metrics
    target_radius = 1e-4
    if target_radius not in radii:
        idx = np.argmin(np.abs(np.array(radii) - target_radius))
        target_radius = radii[idx]
        
    print(f"Layer-wise analysis targeting radius = {target_radius}")
    
    layer_metrics = defaultdict(dict)
    
    for (norm_name, pert_type, layer), items in layer_grouped.items():
        filtered = [i for i in items if i["radius"] == target_radius]
        if not filtered:
            continue
            
        emp_norms = [i["emp_norm"] for i in filtered]
        jvp_norms = [i["jvp_norm"] for i in filtered]
        theory_norms = [i["theory_weighted_norm"] for i in filtered]
        
        ratios_jvp = [i["jvp_norm"] / target_radius for i in filtered]
        ratios_emp = [i["emp_norm"] / target_radius for i in filtered]
        ratios_theory = [i["theory_weighted_norm"] / target_radius for i in filtered]
        
        cos_sims_emp_x = [i["cos_sim_emp_x"] for i in filtered]
        cos_sims_jvp_x = [i["cos_sim_jvp_x"] for i in filtered]
        
        layer_metrics[f"{norm_name}_{pert_type}"][layer] = {
            "jvp_ratio": get_metrics_and_errors(ratios_jvp),
            "emp_ratio": get_metrics_and_errors(ratios_emp),
            "theory_ratio": get_metrics_and_errors(ratios_theory),
            "cos_sim_emp_x": get_metrics_and_errors(cos_sims_emp_x),
            "cos_sim_jvp_x": get_metrics_and_errors(cos_sims_jvp_x)
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
    
    print("\n--- Summary of Power Law Exponents ---")
    for combo_key, pl in power_laws.items():
        exp_val = pl['exponent']
        exp_str = f"{exp_val:.4f}" if exp_val is not None else "N/A (insufficient data)"
        print(f"  {combo_key}: Exponent = {exp_str} (expected: ~1.0000)")

if __name__ == "__main__":
    main()
