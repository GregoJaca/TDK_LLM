import os
import yaml
import pickle
import numpy as np
import matplotlib.pyplot as plt

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def find_best_linear_regime(x_raw, y_raw, min_points=5):
    """
    Finds the contiguous subset of (x_raw, y_raw) of length >= min_points
    that has the best linear fit on the log-log scale with a slope closest to 1.0
    (preferably between 0.97 and 1.03).
    """
    x_raw = np.array(x_raw)
    y_raw = np.array(y_raw)
    
    # Sort by x
    sort_idx = np.argsort(x_raw)
    x_raw = x_raw[sort_idx]
    y_raw = y_raw[sort_idx]
    
    # Keep only positive values for log-log
    valid = (x_raw > 0) & (y_raw > 0)
    x_clean = x_raw[valid]
    y_clean = y_raw[valid]
    
    n = len(x_clean)
    if n < min_points:
        if n >= 2:
            slope, intercept = np.polyfit(np.log10(x_clean), np.log10(y_clean), 1)
            return float(slope), float(intercept), list(x_clean)
        else:
            return 1.0, 0.0, list(x_raw)
            
    log_x = np.log10(x_clean)
    log_y = np.log10(y_clean)
    
    best_subset = None
    best_score = float('inf')
    best_slope = None
    best_intercept = None
    
    for length in range(min_points, n + 1):
        for start in range(n - length + 1):
            sub_x = log_x[start:start+length]
            sub_y = log_y[start:start+length]
            
            slope, intercept = np.polyfit(sub_x, sub_y, 1)
            preds = slope * sub_x + intercept
            mse = np.mean((sub_y - preds) ** 2)
            slope_deviation = abs(slope - 1.0)
            
            # Score penalty: prefer slopes in [0.97, 1.03]
            if 0.97 <= slope <= 1.03:
                score = mse + 0.01 * slope_deviation
            else:
                score = 1.0 + slope_deviation + mse
                
            if score < best_score:
                best_score = score
                best_slope = float(slope)
                best_intercept = float(intercept)
                best_subset = x_clean[start:start+length]
                
    return best_slope, best_intercept, list(best_subset)

def main():
    config_path = "jacobian_config.yaml"
    if not os.path.exists(config_path):
        print(f"Error: {config_path} not found.")
        return
        
    config = load_config(config_path)
    plotting_cfg = config.get("plotting", {})
    norm_config = config.get("normalization", {})
    results_dir = norm_config.get("results_dir", "./results_normalization")
    
    analyzed_path = os.path.join(results_dir, "analyzed_normalization_data.pkl")
    if not os.path.exists(analyzed_path):
        print(f"Error: Analyzed data {analyzed_path} not found. Run analyze_normalization_jacobians.py first.")
        return
        
    with open(analyzed_path, "rb") as f:
        analysis_results = pickle.load(f)
        
    num_layers = analysis_results["num_layers"]
    radii = analysis_results["radii"]
    target_radius = analysis_results["target_radius"]
    aggregated_sweep = analysis_results["aggregated_sweep"]
    power_laws = analysis_results["power_laws"]
    layer_metrics = analysis_results["layer_metrics"]
    
    plots_dir = os.path.join(results_dir, "aggregated_plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Configure global plotting parameters
    dpi = plotting_cfg.get("dpi", 300)
    font_size = plotting_cfg.get("font_size", 20)
    label_size = plotting_cfg.get("label_size", 20)
    tick_size = plotting_cfg.get("tick_size", 16)
    legend_size = plotting_cfg.get("legend_size", 16)
    error_bars = plotting_cfg.get("error_bars", "fan")
    
    # Get active aggregation metrics to plot
    metric_list = plotting_cfg.get("metric", ["mean", "harmonic"])
    if isinstance(metric_list, str):
        metric_list = [metric_list]
        
    plt.rcParams.update({
        "figure.dpi": dpi,
        "savefig.dpi": dpi,
        "font.size": font_size,
        "axes.labelsize": label_size,
        "xtick.labelsize": tick_size,
        "ytick.labelsize": tick_size,
        "legend.fontsize": legend_size,
        "font.family": "DejaVu Serif",
    })
    
    # Configure parameter versions for sweep and annihilation plots
    versions_to_plot_sweep = []
    if norm_config.get("plot_together_orthogonal_radial", True):
        versions_to_plot_sweep.append(("together", [
            ("orthogonal", "navy", "o", "Orthogonal"), 
            ("radial", "crimson", "s", "Radial")
        ], ""))
    if norm_config.get("plot_orthogonal", True):
        versions_to_plot_sweep.append(("orthogonal", [
            ("orthogonal", "navy", "o", "Orthogonal")
        ], "_orthogonal"))
    if norm_config.get("plot_radial", False):
        versions_to_plot_sweep.append(("radial", [
            ("radial", "crimson", "s", "Radial")
        ], "_radial"))

    versions_to_plot_ann = []
    if norm_config.get("plot_together_orthogonal_radial", True):
        versions_to_plot_ann.append(("together", [
            ("orthogonal", "navy", "o", "Orthogonal Perturbation"), 
            ("radial", "crimson", "s", "Radial Perturbation")
        ], ""))
    if norm_config.get("plot_orthogonal", True):
        versions_to_plot_ann.append(("orthogonal", [
            ("orthogonal", "navy", "o", "Orthogonal Perturbation")
        ], "_orthogonal"))
    if norm_config.get("plot_radial", False):
        versions_to_plot_ann.append(("radial", [
            ("radial", "crimson", "s", "Radial Perturbation")
        ], "_radial"))
        
    for metric_name in metric_list:
        print(f"Generating plots for aggregation metric: {metric_name} (error bars: {error_bars})...")
        
        # 1. Figure 1: Linearity Sweep (Log-Log Plot)
        for norm_name in ["rms_actual", "rms_pure", "layernorm_pure"]:
            # Find active combinations for this norm name
            has_orth = any(k[0] == norm_name and k[1] == "orthogonal" for k in aggregated_sweep.keys())
            if not has_orth:
                continue
                
            for version_name, perts, filename_suffix in versions_to_plot_sweep:
                plt.figure(figsize=(10, 8))
                sweep_radii = sorted(radii)
                
                for pert_type, color, marker, label_name in perts:
                    emp_vals = []
                    jvp_vals = []
                    theory_vals = []
                    
                    # Percentiles for fan shading
                    p10_vals, p25_vals, p75_vals, p90_vals = [], [], [], []
                    # Error values for standard deviation propagation
                    err_vals = []
                    
                    # Identify proper error propagation key
                    err_key = "std" if metric_name == "mean" else "harmonic_std"
                    
                    for r in sweep_radii:
                        key = (norm_name, pert_type, r)
                        if key in aggregated_sweep:
                            item = aggregated_sweep[key]
                            emp_vals.append(item["emp_norm"][metric_name])
                            jvp_vals.append(item["jvp_norm"][metric_name])
                            theory_vals.append(item["theory_norm"][metric_name])
                            
                            p10_vals.append(item["emp_norm"]["p10"])
                            p25_vals.append(item["emp_norm"]["p25"])
                            p75_vals.append(item["emp_norm"]["p75"])
                            p90_vals.append(item["emp_norm"]["p90"])
                            
                            err_vals.append(item["emp_norm"][err_key])
                        else:
                            emp_vals.append(np.nan)
                            jvp_vals.append(np.nan)
                            theory_vals.append(np.nan)
                            p10_vals.append(np.nan)
                            p25_vals.append(np.nan)
                            p75_vals.append(np.nan)
                            p90_vals.append(np.nan)
                            err_vals.append(np.nan)
                            
                    emp_vals = np.array(emp_vals)
                    jvp_vals = np.array(jvp_vals)
                    theory_vals = np.array(theory_vals)
                    p10_vals = np.array(p10_vals)
                    p25_vals = np.array(p25_vals)
                    p75_vals = np.array(p75_vals)
                    p90_vals = np.array(p90_vals)
                    err_vals = np.array(err_vals)
                    
                    combo_key = f"{norm_name}_{pert_type}_{metric_name}"
                    exponent = power_laws.get(combo_key, {}).get("exponent", None)
                    exp_label = f" (k={exponent:.3f})" if exponent is not None else ""
                    if exponent is not None:
                        print(f"Sweep Plot: Norm={norm_name}, Perturbation={pert_type}, Metric={metric_name} | Slope of fit = {exponent:.6f}")
                    
                    # Plot empirical finite difference
                    plt.plot(sweep_radii, emp_vals, marker=marker, color=color, linewidth=2.5, 
                             label=f"Empirical {label_name}{exp_label}")
                    
                    # Proper error propagation shading
                    valid_mask = ~np.isnan(emp_vals)
                    if error_bars in ["fan", "percentiles"]:
                        valid_fan = ~np.isnan(p10_vals) & ~np.isnan(p90_vals)
                        eps_safe = 1e-15
                        p10_clean = np.maximum(eps_safe, np.array(p10_vals))
                        p25_clean = np.maximum(eps_safe, np.array(p25_vals))
                        p75_clean = np.maximum(eps_safe, np.array(p75_vals))
                        p90_clean = np.maximum(eps_safe, np.array(p90_vals))
                        plt.fill_between(np.array(sweep_radii)[valid_fan], p10_clean[valid_fan], p90_clean[valid_fan], 
                                         color=color, alpha=0.1)
                        plt.fill_between(np.array(sweep_radii)[valid_fan], p25_clean[valid_fan], p75_clean[valid_fan], 
                                         color=color, alpha=0.2)
                    elif error_bars == "std":
                        valid_err = valid_mask & ~np.isnan(err_vals)
                        eps_safe = 1e-15
                        with np.errstate(divide='ignore', invalid='ignore'):
                            std_log = err_vals / np.maximum(emp_vals, eps_safe)
                            lower = emp_vals * np.exp(-std_log)
                            upper = emp_vals * np.exp(std_log)
                        lower = np.maximum(eps_safe, np.where(np.isnan(lower) | np.isinf(lower), eps_safe, lower))
                        upper = np.where(np.isnan(upper) | np.isinf(upper), emp_vals, upper)
                        plt.fill_between(np.array(sweep_radii)[valid_err], lower[valid_err], upper[valid_err], 
                                         color=color, alpha=0.15)
                                     
                    # Plot JVP (Option 2 - Linear baseline)
                    plt.plot(sweep_radii, jvp_vals, linestyle="--", color=color, alpha=0.6, 
                             label=f"JVP {label_name}")
                             
                plt.xscale("log")
                plt.yscale("log")
                plt.xlabel(r"Perturbation Magnitude")
                plt.ylabel(r"Output Perturbation")
                plt.grid(True, which="both", alpha=0.3)
                plt.legend(loc="lower right")
                plt.tight_layout()
                
                plot_name = f"normalization_sweep_{norm_name}_{metric_name}{filename_suffix}.png"
                plt.savefig(os.path.join(plots_dir, plot_name), dpi=dpi)
                plt.close()
                print(f"Saved linearity sweep plot to {os.path.join(plots_dir, plot_name)}")
            
        # 2. Figure 2: Radial Scaling Verification (Ratio vs Layer)
        for norm_name in ["rms_actual", "rms_pure", "layernorm_pure"]:
            combo_key = f"{norm_name}_orthogonal"
            if combo_key not in layer_metrics:
                continue
                
            metrics_dict = layer_metrics[combo_key]
            sorted_layers = sorted(metrics_dict.keys())
            layers_arr = np.array(sorted_layers)
            
            theory_vals = []
            jvp_vals = []
            emp_vals = []
            
            jvp_p10, jvp_p25, jvp_p75, jvp_p90 = [], [], [], []
            jvp_errs = []
            err_key = "std" if metric_name == "mean" else "harmonic_std"
            
            for l in sorted_layers:
                stats = metrics_dict[l]
                theory_vals.append(stats["theory_ratio"][metric_name])
                jvp_vals.append(stats["jvp_ratio"][metric_name])
                emp_vals.append(stats["emp_ratio"][metric_name])
                
                jvp_p10.append(stats["jvp_ratio"]["p10"])
                jvp_p25.append(stats["jvp_ratio"]["p25"])
                jvp_p75.append(stats["jvp_ratio"]["p75"])
                jvp_p90.append(stats["jvp_ratio"]["p90"])
                jvp_errs.append(stats["jvp_ratio"][err_key])
                
            theory_vals = np.array(theory_vals)
            jvp_vals = np.array(jvp_vals)
            emp_vals = np.array(emp_vals)
            jvp_p10 = np.array(jvp_p10)
            jvp_p25 = np.array(jvp_p25)
            jvp_p75 = np.array(jvp_p75)
            jvp_p90 = np.array(jvp_p90)
            jvp_errs = np.array(jvp_errs)
            
            plt.figure(figsize=(10, 6))
            
            # Plot theoretical 1/S (or scale/S)
            plt.plot(layers_arr, theory_vals, marker="^", color="crimson", linewidth=2.5, 
                     label=r"Analytical Theory $1/S$")
                     
            # Plot JVP ratio
            plt.plot(layers_arr, jvp_vals, marker="o", color="navy", linewidth=2, linestyle="--", 
                     label="JVP Scaling Ratio")
            
            # Error propagation shading
            if error_bars in ["fan", "percentiles"]:
                plt.fill_between(layers_arr, jvp_p10, jvp_p90, color="navy", alpha=0.1)
                plt.fill_between(layers_arr, jvp_p25, jvp_p75, color="navy", alpha=0.2)
            elif error_bars == "std":
                lower = np.maximum(0.0, jvp_vals - jvp_errs)
                plt.fill_between(layers_arr, lower, jvp_vals + jvp_errs, color="navy", alpha=0.15)
            
            # Plot Empirical ratio
            plt.plot(layers_arr, emp_vals, marker="s", color="darkorange", linewidth=1.5, linestyle=":", 
                     label=fr"Empirical Scaling Ratio ($\epsilon$={target_radius:.1e})")
                     
            plt.xlabel("Layer Index")
            plt.ylabel(r"Scaling Gain")
            plt.grid(True, alpha=0.3)
            plt.legend(loc="best")
            plt.tight_layout()
            
            plot_name = f"normalization_scaling_verification_{norm_name}_{metric_name}.png"
            plt.savefig(os.path.join(plots_dir, plot_name), dpi=dpi)
            plt.close()
            print(f"Saved radial scaling verification to {os.path.join(plots_dir, plot_name)}")
            
        # 3. Figure 3: Orthogonal Annihilation Verification (Cosine Similarity vs Layer)
        ann_yscale = norm_config.get("annihilation_yscale", "linear")
        for norm_name in ["rms_actual", "rms_pure", "layernorm_pure"]:
            for version_name, perts, filename_suffix in versions_to_plot_ann:
                plt.figure(figsize=(10, 6))
                
                plotted = False
                for pert_type, color, marker, label_name in perts:
                    combo_key = f"{norm_name}_{pert_type}"
                    if combo_key not in layer_metrics:
                        continue
                        
                    metrics_dict = layer_metrics[combo_key]
                    sorted_layers = sorted(metrics_dict.keys())
                    layers_arr = np.array(sorted_layers)
                    
                    cos_sim_vals = []
                    p10_vals, p25_vals, p75_vals, p90_vals = [], [], [], []
                    err_vals = []
                    
                    err_key = "std" if metric_name == "mean" else "harmonic_std"
                    
                    for l in sorted_layers:
                        stats = metrics_dict[l]
                        cos_sim_vals.append(stats["cos_sim_jvp_x"][metric_name])
                        
                        p10_vals.append(stats["cos_sim_jvp_x"]["p10"])
                        p25_vals.append(stats["cos_sim_jvp_x"]["p25"])
                        p75_vals.append(stats["cos_sim_jvp_x"]["p75"])
                        p90_vals.append(stats["cos_sim_jvp_x"]["p90"])
                        err_vals.append(stats["cos_sim_jvp_x"][err_key])
                        
                    cos_sim_vals = np.array(cos_sim_vals)
                    p10_vals = np.array(p10_vals)
                    p25_vals = np.array(p25_vals)
                    p75_vals = np.array(p75_vals)
                    p90_vals = np.array(p90_vals)
                    err_vals = np.array(err_vals)
                    
                    plt.plot(layers_arr, cos_sim_vals, marker=marker, color=color, linewidth=2.5, 
                             label=label_name)
                    
                    if error_bars in ["fan", "percentiles"]:
                        plt.fill_between(layers_arr, p10_vals, p90_vals, color=color, alpha=0.1)
                        plt.fill_between(layers_arr, p25_vals, p75_vals, color=color, alpha=0.2)
                    elif error_bars == "std":
                        plt.fill_between(layers_arr, cos_sim_vals - err_vals, cos_sim_vals + err_vals, color=color, alpha=0.15)
                        
                    plotted = True
                    
                if not plotted:
                    plt.close()
                    continue
                    
                plt.axhline(0.0, color="black", linestyle="-", linewidth=1.5)
                
                plt.yscale(ann_yscale)
                plt.xlabel("Layer Index")
                plt.ylabel(r"Survival")
                plt.grid(True, alpha=0.3)
                plt.legend(loc="best")
                plt.tight_layout()
                
                plot_name = f"normalization_orthogonal_annihilation_{norm_name}_{metric_name}{filename_suffix}.png"
                plt.savefig(os.path.join(plots_dir, plot_name), dpi=dpi)
                plt.close()
                print(f"Saved orthogonal annihilation verification to {os.path.join(plots_dir, plot_name)}")

    # 4. Extract and plot cumulative and layer Jacobians from linear normalization sweep data
    plot_extracted_normalization_jacobians(analysis_results, plots_dir, plotting_cfg, norm_config)

def plot_extracted_normalization_jacobians(analysis_results, plots_dir, plotting_cfg, norm_config):
    """
    Extracts the cumulative Jacobian A(l) and the layer-by-layer Jacobian J(l)
    from the linear regime of the normalization sweep data, and plots them.
    Also fits and saves layer-wise slopes and intercepts to a CSV.
    """
    plot_cum = plotting_cfg.get("plot_extracted_cumulative_jacobian", False)
    plot_lay = plotting_cfg.get("plot_extracted_layer_jacobian", False)
    plot_tog = plotting_cfg.get("plot_extracted_jacobians_together", True)
    
    if not (plot_cum or plot_lay or plot_tog):
        return
        
    layer_sweep_data = analysis_results.get("layer_sweep_data", {})
    if not layer_sweep_data:
        print("No layer sweep data found. Cannot extract Jacobians.")
        return
        
    radii = analysis_results["radii"]
    sorted_radii = sorted(radii)
    
    # Identify linear regime: 1e-7 <= r <= 1e-2
    linear_radii = [r for r in sorted_radii if 1e-7 <= r <= 1e-2]
    if len(linear_radii) < 2:
        linear_radii = [r for r in sorted_radii if r <= 1e-2]
        
    metric_list = plotting_cfg.get("metric", ["mean", "harmonic"])
    if isinstance(metric_list, str):
        metric_list = [metric_list]
        
    x_scales = plotting_cfg.get("x_scales", ["linear"])
    y_scales = plotting_cfg.get("y_scales", ["log"])
    if isinstance(x_scales, str):
        x_scales = [x_scales]
    if isinstance(y_scales, str):
        y_scales = [y_scales]
        
    error_bars = plotting_cfg.get("error_bars", "fan")
    dpi = plotting_cfg.get("dpi", 300)
    
    import csv
    
    for group_key, radii_data in layer_sweep_data.items():
        info_key = group_key
        
        for current_metric in metric_list:
            # Extract layer array from the first radius
            first_radius = sorted_radii[0]
            if current_metric not in radii_data[first_radius]:
                continue
            layer_arr = np.array(radii_data[first_radius]["layers"])
            num_layers = len(layer_arr)
            
            # Helper to compute gain array for a given key in radii_data
            def compute_gain_for_key(metric_name):
                # The per-layer Jacobian is the gain at each layer:
                J_arr = np.zeros(num_layers)
                for i in range(num_layers):
                    layer_radii = []
                    layer_vals = []
                    for r in sorted_radii:
                        if metric_name in radii_data[r] and len(radii_data[r][metric_name]) > i:
                            layer_radii.append(r)
                            layer_vals.append(radii_data[r][metric_name][i])
                    
                    slope, constant, best_subset = find_best_linear_regime(layer_radii, layer_vals, min_points=5)
                    J_arr[i] = 10**constant if constant is not None else 1.0
                    
                    # Print to terminal the exact slope and fit for every layer
                    if metric_name == current_metric:
                        print(f"Normalization [Extract J] | Group={group_key} | Layer {layer_arr[i]} | Slope (Exponent) = {slope:.6f} | Constant (Log-Gain) = {constant:.6f} | Gain = {10**constant:.6e} | Range = [{min(best_subset):.1e}, {max(best_subset):.1e}]")
                
                # The cumulative Jacobian is the product of per-layer Jacobians:
                A_arr = np.cumprod(J_arr)
                A_with_input = np.insert(A_arr, 0, 1.0)
                
                return A_with_input, J_arr
            
            # Compute main aggregated metric
            A_with_input, layer_jac = compute_gain_for_key(current_metric)
            layer_arr_with_input = np.insert(layer_arr, 0, layer_arr[0] - 1)
            
            # Save results of layer-wise linear fits to a CSV file
            csv_rows = []
            for i in range(num_layers):
                layer_radii = []
                layer_vals = []
                for r in sorted_radii:
                    if current_metric in radii_data[r] and len(radii_data[r][current_metric]) > i:
                        layer_radii.append(r)
                        layer_vals.append(radii_data[r][current_metric][i])
                        
                slope, constant, best_subset = find_best_linear_regime(layer_radii, layer_vals, min_points=5)
                
                csv_rows.append({
                    "Layer": layer_arr[i],
                    "Slope": slope,
                    "Intercept": constant,
                    "Gain (10^Intercept)": 10**constant if constant is not None else None,
                    "Fit Range Start": min(best_subset) if best_subset else None,
                    "Fit Range End": max(best_subset) if best_subset else None
                })
                
            csv_path = os.path.join(plots_dir, f"normalization_scaling_fit_results_{info_key}_{current_metric}.csv")
            with open(csv_path, mode="w", newline="") as f_csv:
                writer = csv.writer(f_csv)
                writer.writerow(["Layer", "Slope (Log-Log)", "Intercept (Log-Log)", "Gain (10^Intercept)", "Fit Range Start", "Fit Range End"])
                for row in csv_rows:
                    writer.writerow([row["Layer"], row["Slope"], row["Intercept"], row["Gain (10^Intercept)"], row["Fit Range Start"], row["Fit Range End"]])
            print(f"Saved layer-wise linear fit results CSV to {csv_path}")
            
            # Compute fan error bounds
            has_fan = (error_bars in ["fan", "percentiles"])
            if has_fan:
                if "p10" in radii_data[first_radius]:
                    A_p10, layer_jac_p10 = compute_gain_for_key("p10")
                    A_p25, layer_jac_p25 = compute_gain_for_key("p25")
                    A_p75, layer_jac_p75 = compute_gain_for_key("p75")
                    A_p90, layer_jac_p90 = compute_gain_for_key("p90")
                else:
                    has_fan = False
            
            # Now plot for each combination of scale
            for x_scale in x_scales:
                for y_scale in y_scales:
                    # 1. Plot Cumulative Aggregate Jacobian
                    if plot_cum:
                        plt.figure(figsize=(8, 6))
                        plt.axhline(1.0, color='gray', linestyle='--', linewidth=1.5, zorder=1)
                        plt.plot(layer_arr_with_input, A_with_input, marker='o', color='b', linewidth=2, markersize=4, zorder=3)
                        
                        if has_fan:
                            plt.fill_between(layer_arr_with_input, np.maximum(1e-12, A_p10), A_p90, color='b', alpha=0.1, zorder=2)
                            plt.fill_between(layer_arr_with_input, np.maximum(1e-12, A_p25), A_p75, color='b', alpha=0.2, zorder=2)
                            
                        plt.xlabel("Layer")
                        plt.ylabel("Linearized Perturbation Gain")
                        plt.xscale(x_scale)
                        plt.yscale(y_scale)
                        plt.grid(True, alpha=0.3, which="both")
                        plt.tight_layout()
                        filename = f"{info_key}_extracted_cumulative_jacobian_{current_metric}_xscale-{x_scale}_yscale-{y_scale}.png"
                        plt.savefig(os.path.join(plots_dir, filename), dpi=dpi)
                        plt.close()
                        print(f"Saved extracted cumulative Jacobian plot to {os.path.join(plots_dir, filename)}")
                        
                    # 2. Plot Layer Jacobian
                    if plot_lay:
                        plt.figure(figsize=(8, 6))
                        plt.axhline(1.0, color='gray', linestyle='--', linewidth=1.5, zorder=1)
                        plt.plot(layer_arr, layer_jac, marker='s', color='r', linewidth=2, markersize=4, zorder=3)
                        
                        if has_fan:
                            plt.fill_between(layer_arr, np.maximum(1e-12, layer_jac_p10), layer_jac_p90, color='r', alpha=0.1, zorder=2)
                            plt.fill_between(layer_arr, np.maximum(1e-12, layer_jac_p25), layer_jac_p75, color='r', alpha=0.2, zorder=2)
                            
                        plt.xlabel("Layer")
                        plt.ylabel("Linearized Perturbation Gain")
                        plt.xscale(x_scale)
                        plt.yscale(y_scale)
                        plt.grid(True, alpha=0.3, which="both")
                        plt.tight_layout()
                        filename = f"{info_key}_extracted_layer_jacobian_{current_metric}_xscale-{x_scale}_yscale-{y_scale}.png"
                        plt.savefig(os.path.join(plots_dir, filename), dpi=dpi)
                        plt.close()
                        print(f"Saved extracted layer Jacobian plot to {os.path.join(plots_dir, filename)}")
                        
                    # 3. Plot both together
                    if plot_tog:
                        plt.figure(figsize=(8, 6))
                        plt.axhline(1.0, color='gray', linestyle='--', linewidth=1.5, zorder=1, label="Baseline")
                        plt.plot(layer_arr_with_input, A_with_input, marker='o', color='b', linewidth=2, markersize=4, label="Aggregate Jacobian", zorder=4)
                        plt.plot(layer_arr, layer_jac, marker='s', color='r', linewidth=2, markersize=4, label="Layer Jacobian", zorder=4)
                        
                        if has_fan:
                            plt.fill_between(layer_arr_with_input, np.maximum(1e-12, A_p10), A_p90, color='b', alpha=0.08, zorder=2)
                            plt.fill_between(layer_arr_with_input, np.maximum(1e-12, A_p25), A_p75, color='b', alpha=0.15, zorder=2)
                            plt.fill_between(layer_arr, np.maximum(1e-12, layer_jac_p10), layer_jac_p90, color='r', alpha=0.08, zorder=2)
                            plt.fill_between(layer_arr, np.maximum(1e-12, layer_jac_p25), layer_jac_p75, color='r', alpha=0.15, zorder=2)
                            
                        plt.xlabel("Layer")
                        plt.ylabel("Linearized Perturbation Gain")
                        plt.xscale(x_scale)
                        plt.yscale(y_scale)
                        plt.grid(True, alpha=0.3, which="both")
                        plt.legend(loc="best")
                        plt.tight_layout()
                        filename = f"{info_key}_extracted_jacobians_together_{current_metric}_xscale-{x_scale}_yscale-{y_scale}.png"
                        plt.savefig(os.path.join(plots_dir, filename), dpi=dpi)
                        plt.close()
                        print(f"Saved extracted Jacobians together plot to {os.path.join(plots_dir, filename)}")

if __name__ == "__main__":
    main()
