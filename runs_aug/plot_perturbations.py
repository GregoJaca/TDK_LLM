import os
import re
import csv
import yaml
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def get_safe_filename_info(group_key, group_titles):
    title = group_titles.get(group_key, group_key)
    setup_match = re.search(r"Setup:\s*([^|]+)", title)
    prompt_match = re.search(r"Prompt:\s*'([^']+)'", title)
    
    setup_name = setup_match.group(1).strip() if setup_match else group_key
    prompt_val = prompt_match.group(1).strip() if prompt_match else ""
    
    setup_clean = re.sub(r'[^a-zA-Z0-9_-]', '_', setup_name).strip('_')
    setup_clean = re.sub(r'_+', '_', setup_clean)
    
    if "Aggregated" in title:
        return f"{setup_clean}_aggregated"
        
    if prompt_val:
        prompt_clean = re.sub(r'[^a-zA-Z0-9_-]', '_', prompt_val).strip('_')
        prompt_clean = re.sub(r'_+', '_', prompt_clean)[:30].strip('_')
        return f"{setup_clean}_{prompt_clean}"
        
    return re.sub(r'[^a-zA-Z0-9_-]', '_', group_key).strip('_')

def plot_perturbations(base_results_dir, plotting_cfg):
    data_path = os.path.join(base_results_dir, "analyzed_data.pkl")
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Run analyze_perturbations.py first.")
        return
        
    with open(data_path, "rb") as f:
        analyzed_data = pickle.load(f)
        
    group_titles = analyzed_data["group_titles"]
    data = analyzed_data["data"]
    
    plots_dir = os.path.join(base_results_dir, "aggregated_plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Expose and apply global styling configuration
    dpi = plotting_cfg.get("dpi", 300)
    font_size = plotting_cfg.get("font_size", 14)
    label_size = plotting_cfg.get("label_size", 14)
    tick_size = plotting_cfg.get("tick_size", 12)
    legend_size = plotting_cfg.get("legend_size", 11)
    
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
    
    metric_list = plotting_cfg.get("metric", ["mean"])
    if isinstance(metric_list, str):
        metric_list = [metric_list]
        
    error_bars = plotting_cfg.get("error_bars", "none")
    plot_prompt_together = plotting_cfg.get("plot_prompt_together", False)
    plot_prompt_separated = plotting_cfg.get("plot_prompt_separated", True)
    separate_figure_metrics = plotting_cfg.get("separate_figure_metrics", False)
    plot_magnitude_together = plotting_cfg.get("plot_magnitude_together", True)
    plot_magnitude_separated = plotting_cfg.get("plot_magnitude_separated", True)
    x_scales = plotting_cfg.get("x_scales", plotting_cfg.get("x_scale", ["linear"]))
    y_scales = plotting_cfg.get("y_scales", plotting_cfg.get("y_scale", ["linear"]))
    if isinstance(x_scales, str):
        x_scales = [x_scales]
    if isinstance(y_scales, str):
        y_scales = [y_scales]
    
    # 1. Standard Divergence Plots with Fan Shading
    metrics_to_process = [[m] for m in metric_list] if separate_figure_metrics else [metric_list]
    
    for x_scale in x_scales:
        for y_scale in y_scales:
            for current_metric_list in metrics_to_process:
                for group_key, radii_data in data.items():
                    if group_key.endswith("_aggregated") and not plot_prompt_together:
                        continue
                    if not group_key.endswith("_aggregated") and not plot_prompt_separated:
                        continue
                        
                    sorted_radii = sorted(radii_data.keys())
                    info_key = get_safe_filename_info(group_key, group_titles)
                    
                    # --- Option A: Plot all magnitudes together ---
                    if plot_magnitude_together:
                        # STRICT FILTER: Only plot if metric is 'harmonic' and scale is linear-log
                        # Target plot: single_token_perturb_all_All_Prompts_Aggregated_aggregated_metric-harmonic_err-fan_xscale-linear_yscale-log.png
                        if "harmonic" not in current_metric_list or x_scale != "linear" or y_scale != "log":
                            continue
                            
                        plt.figure(figsize=(10, 6))
                        combinations = []
                        for r in sorted_radii:
                            for m in current_metric_list:
                                combinations.append((r, m))
                                
                        colors = plt.cm.tab10(np.linspace(0, 1, max(len(combinations), 10)))
                        
                        for i, (radius, current_metric) in enumerate(combinations):
                            color = colors[i % len(colors)]
                            metrics_dict = radii_data[radius]
                            
                            layer_arr = metrics_dict["layers"]
                            m_arr = metrics_dict[current_metric]
                            
                            line_label = f"R={radius} | {current_metric.capitalize()}" if len(current_metric_list) > 1 else f"{radius}"
                            plt.plot(layer_arr, m_arr, marker='o', color=color, label=line_label, linewidth=2, markersize=4)
                            
                            # Fan Shading based on percentiles if available
                            if error_bars == "fan" or error_bars == "percentiles":
                                if "p10" in metrics_dict and "p90" in metrics_dict:
                                    lower_p10 = metrics_dict["p10"]
                                    if y_scale == "log":
                                        lower_p10 = np.maximum(1e-12, lower_p10)
                                    plt.fill_between(layer_arr, lower_p10, metrics_dict["p90"], color=color, alpha=0.1, label='_nolegend_')
                                if "p25" in metrics_dict and "p75" in metrics_dict:
                                    lower_p25 = metrics_dict["p25"]
                                    if y_scale == "log":
                                        lower_p25 = np.maximum(1e-12, lower_p25)
                                    plt.fill_between(layer_arr, lower_p25, metrics_dict["p75"], color=color, alpha=0.2, label='_nolegend_')
                            else:
                                err_key = f"{current_metric}_{error_bars}"
                                if err_key in metrics_dict:
                                    e_arr = metrics_dict[err_key]
                                elif error_bars in metrics_dict:
                                    e_arr = metrics_dict[error_bars]
                                else:
                                    e_arr = None
                                    
                                if e_arr is not None:
                                    lower = np.maximum(0, m_arr - e_arr) if error_bars in ["std", "var"] else m_arr - e_arr
                                    if y_scale == "log":
                                        lower = np.maximum(1e-12, lower)
                                    plt.fill_between(layer_arr, lower, m_arr + e_arr, color=color, alpha=0.2, label='_nolegend_')
                                
                        if plotting_cfg.get("show_title", False):
                            title = group_titles[group_key]
                            if separate_figure_metrics and len(current_metric_list) == 1:
                                title += f" ({current_metric_list[0].capitalize()})"
                            plt.title(f"Divergence over Layers | {title}")
                            
                        plt.xlabel("Layer")
                        plt.ylabel("Distance")
                        plt.xscale(x_scale)
                        plt.yscale(y_scale)
                        plt.grid(True, alpha=0.3)
                        
                        if len(combinations) > 1:
                            plt.legend(title="Perturbation", loc='upper left', bbox_to_anchor=(1, 1))
                        plt.tight_layout()
                        
                        metric_str = "-".join(current_metric_list)
                        plot_filename = f"{info_key}_metric-{metric_str}_err-{error_bars}_xscale-{x_scale}_yscale-{y_scale}.png"
                        plt.savefig(os.path.join(plots_dir, plot_filename), dpi=dpi)
                        plt.close()
        
                    # --- Option B: Plot each magnitude separately ---
                    if plot_magnitude_separated:
                        # STRICT FILTER: skip all since plot_magnitude_separated is false
                        continue
 
    # 2. Distribution Heatmaps (One per Radius per Group)
    if plotting_cfg.get("plot_heatmap", True):
        pass # Skipped by config
 
    # 3. Average over Layers 2-25 vs Perturbation Magnitude Plot (using robust error and config scales)
    if plotting_cfg.get("plot_scaling_law", True):
        for x_scale in x_scales:
            for y_scale in y_scales:
                for current_metric in metric_list:
                    for group_key, radii_data in data.items():
                        if group_key.endswith("_aggregated") and not plot_prompt_together:
                            continue
                        if not group_key.endswith("_aggregated") and not plot_prompt_separated:
                            continue
            
                        sorted_radii = sorted(radii_data.keys())
                        x_radii = []
                        y_avg = []
                        y_err_up = []
                        y_err_low = []
            
                        for radius in sorted_radii:
                            metrics_dict = radii_data[radius]
                            layer_arr = metrics_dict["layers"]
                            m_arr = metrics_dict[current_metric]
                            mask = (layer_arr >= 2) & (layer_arr <= 25)
                            if not np.any(mask): continue
                                
                            selected_vals = m_arr[mask]
                            x_radii.append(radius)
                            y_avg.append(np.mean(selected_vals))
                            
                            metric_var_key = f"{current_metric}_var" if f"{current_metric}_var" in metrics_dict else "var"
                            
                            if error_bars in ["std", "var"]:
                                var_arr = metrics_dict[metric_var_key]
                                overall_var = np.mean(var_arr[mask]) + np.var(selected_vals)
                                err = np.sqrt(overall_var) if error_bars == "std" else overall_var
                                y_err_low.append(err)
                                y_err_up.append(err)
                            elif error_bars == "fan" or error_bars == "percentiles":
                                low_spread = np.mean(metrics_dict["p10"][mask])
                                high_spread = np.mean(metrics_dict["p90"][mask])
                                y_err_low.append(np.mean(selected_vals) - low_spread)
                                y_err_up.append(high_spread - np.mean(selected_vals))
                            else:
                                y_err_low.append(0); y_err_up.append(0)
            
                        if not x_radii: continue
                        info_key = get_safe_filename_info(group_key, group_titles)
            
                        # --- Regular Scaling Law Plot ---
                        # STRICT FILTER: Skip regular scaling plot if scaling_prompts_separate is true
                        if not plotting_cfg.get("scaling_prompts_separate", False):
                            plt.figure(figsize=(8, 6))
                            if any(y_err_up):
                                low_err = np.clip(y_err_low, 0, np.array(y_avg) * 0.99)
                                plt.errorbar(x_radii, y_avg, yerr=[low_err, y_err_up], marker='o', capsize=5, color='b')
                            else:
                                plt.plot(x_radii, y_avg, marker='o', color='b')
                
                            plt.xscale(x_scale)
                            plt.yscale(y_scale)
                            if plotting_cfg.get("show_title", False):
                                plt.title(f"Scaling Law (L2-25) | {group_titles[group_key]}")
                            plt.xlabel("Perturbation")
                            plt.ylabel("Average Distance")
                            plt.grid(True, alpha=0.3, which="both")
                            plt.tight_layout()
                            plt.savefig(os.path.join(plots_dir, f"{info_key}_scaling_{current_metric}_xscale-{x_scale}_yscale-{y_scale}.png"), dpi=dpi)
                            plt.close()

                        # --- Rainbow Scaling Law Plot (Individual Prompt Trajectories) ---
                        if plotting_cfg.get("scaling_prompts_separate", False) and group_key.endswith("_aggregated"):
                            # STRICT FILTER: Only plot median in log-log scale
                            # Target plot: single_token_perturb_all_All_Prompts_Aggregated_aggregated_scaling_rainbow_median_xscale-log_yscale-log.png
                            if current_metric != "median" or x_scale != "log" or y_scale != "log":
                                continue
                                
                            setup_name = group_key.replace("_aggregated", "")
                            prompt_keys = [k for k in data.keys() if k.startswith(setup_name) and not k.endswith("_aggregated")]
                            
                            if prompt_keys:
                                plt.figure(figsize=(8, 6))
                                cmap = plt.cm.rainbow(np.linspace(0, 1, len(prompt_keys)))
                                
                                fit_rows = []
                                individual_slopes = []
                                
                                # Fit and save curves for individual prompts
                                for p_idx, p_key in enumerate(prompt_keys):
                                    p_radii_data = data[p_key]
                                    p_sorted_radii = sorted(p_radii_data.keys())
                                    px_radii = []
                                    py_avg = []
                                    
                                    for radius in p_sorted_radii:
                                        p_metrics_dict = p_radii_data[radius]
                                        p_layer_arr = p_metrics_dict["layers"]
                                        p_m_arr = p_metrics_dict[current_metric]
                                        p_mask = (p_layer_arr >= 2) & (p_layer_arr <= 25)
                                        if not np.any(p_mask): continue
                                        
                                        px_radii.append(radius)
                                        py_avg.append(np.mean(p_m_arr[p_mask]))
                                        
                                    if px_radii:
                                        # Render curve
                                        plt.plot(px_radii, py_avg, color=cmap[p_idx], alpha=0.3, linewidth=1.5, linestyle='-')
                                        
                                        # Linear Fit on the last 8 values (or all if fewer than 8)
                                        px_arr = np.array(px_radii)
                                        py_arr = np.array(py_avg)
                                        x_fit = px_arr[-8:]
                                        y_fit = py_arr[-8:]
                                        
                                        # Avoid log(<=0)
                                        valid = (x_fit > 0) & (y_fit > 0)
                                        if np.sum(valid) >= 2:
                                            log_x = np.log10(x_fit[valid])
                                            log_y = np.log10(y_fit[valid])
                                            slope, constant = np.polyfit(log_x, log_y, 1)
                                            individual_slopes.append(slope)
                                        else:
                                            slope, constant = None, None
                                            
                                        smallest_pert_dist = py_arr[0] if len(py_arr) > 0 else None
                                        fit_rows.append({
                                            "Curve": f"Prompt {p_idx}",
                                            "Slope": slope,
                                            "Constant": constant,
                                            "Smallest_Perturbation_Distance": smallest_pert_dist
                                        })
                                
                                # Fit and save curves for aggregated mean
                                ax_arr = np.array(x_radii)
                                ay_arr = np.array(y_avg)
                                x_fit = ax_arr[-8:]
                                y_fit = ay_arr[-8:]
                                
                                valid = (x_fit > 0) & (y_fit > 0)
                                if np.sum(valid) >= 2:
                                    log_x = np.log10(x_fit[valid])
                                    log_y = np.log10(y_fit[valid])
                                    slope, constant = np.polyfit(log_x, log_y, 1)
                                else:
                                    slope, constant = None, None
                                    
                                smallest_pert_dist = ay_arr[0] if len(ay_arr) > 0 else None
                                fit_rows.append({
                                    "Curve": "Aggregated Mean",
                                    "Slope": slope,
                                    "Constant": constant,
                                    "Smallest_Perturbation_Distance": smallest_pert_dist
                                })
                                
                                # Calculate stats on individual slopes
                                avg_slope = np.mean(individual_slopes) if individual_slopes else None
                                std_slope = np.std(individual_slopes) if individual_slopes else None
                                
                                # Save results to CSV
                                csv_path = os.path.join(plots_dir, f"scaling_fit_results_{info_key}_{current_metric}.csv")
                                with open(csv_path, mode="w", newline="") as f_csv:
                                    writer = csv.writer(f_csv)
                                    writer.writerow(["Curve", "Slope", "Constant", "Smallest_Perturbation_Distance"])
                                    for row in fit_rows:
                                        writer.writerow([row["Curve"], row["Slope"], row["Constant"], row["Smallest_Perturbation_Distance"]])
                                    writer.writerow([])
                                    writer.writerow(["Average of Individual Slopes", avg_slope, "", ""])
                                    writer.writerow(["Std of Individual Slopes", std_slope, "", ""])
                                
                                print(f"Saved linear fit results CSV to {csv_path}")
                                
                                # Overlay aggregated mean curve
                                plt.plot(x_radii, y_avg, marker='o', color='black', linewidth=3, markersize=6)
                                
                                plt.xscale(x_scale)
                                plt.yscale(y_scale)
                                if plotting_cfg.get("show_title", False):
                                    plt.title(f"Scaling Law with Prompt Trajectories | {group_titles[group_key]}")
                                plt.xlabel("Perturbation")
                                plt.ylabel("Average Distance")
                                plt.grid(True, alpha=0.3, which="both")
                                
                                from matplotlib.lines import Line2D
                                legend_elements = [
                                    Line2D([0], [0], color='black', linewidth=3, marker='o', label='Aggregated Mean'),
                                    Line2D([0], [0], color='gray', linewidth=1.5, alpha=0.5, label='Individual Prompts')
                                ]
                                plt.legend(handles=legend_elements, loc='upper left')
                                plt.tight_layout()
                                
                                plt.savefig(os.path.join(plots_dir, f"{info_key}_scaling_rainbow_{current_metric}_xscale-{x_scale}_yscale-{y_scale}.png"), dpi=dpi)
                                plt.close()

    # Extract and plot cumulative and layer Jacobians from linear perturbation data
    plot_extracted_jacobians(data, group_titles, plots_dir, plotting_cfg)

def plot_extracted_jacobians(data, group_titles, plots_dir, plotting_cfg):
    """
    Extracts the cumulative Jacobian A(l) and the layer-by-layer Jacobian J(l -> l+1)
    from the linear regime of the perturbation propagation data, and plots them.
    Supports fan error shading and individual prompt swarms.
    """
    plot_cum = plotting_cfg.get("plot_extracted_cumulative_jacobian", False)
    plot_lay = plotting_cfg.get("plot_extracted_layer_jacobian", False)
    plot_tog = plotting_cfg.get("plot_extracted_jacobians_together", False)
    
    if not (plot_cum or plot_lay or plot_tog):
        return
        
    metric_list = plotting_cfg.get("metric", ["mean"])
    if isinstance(metric_list, str):
        metric_list = [metric_list]
        
    x_scales = plotting_cfg.get("x_scales", ["linear"])
    y_scales = plotting_cfg.get("y_scales", ["log"])
    if isinstance(x_scales, str):
        x_scales = [x_scales]
    if isinstance(y_scales, str):
        y_scales = [y_scales]
        
    plot_prompt_together = plotting_cfg.get("plot_prompt_together", False)
    plot_prompt_separated = plotting_cfg.get("plot_prompt_separated", True)
    error_bars = plotting_cfg.get("error_bars", "none")
    dpi = plotting_cfg.get("dpi", 300)
    
    for group_key, radii_data in data.items():
        if group_key.endswith("_aggregated") and not plot_prompt_together:
            continue
        if not group_key.endswith("_aggregated") and not plot_prompt_separated:
            continue
            
        sorted_radii = sorted(radii_data.keys())
        if not sorted_radii:
            continue
            
        info_key = get_safe_filename_info(group_key, group_titles)
        setup_name = group_key.replace("_aggregated", "")
        prompt_keys = [k for k in data.keys() if k.startswith(setup_name) and not k.endswith("_aggregated")]
        
        # Identify the linear regime: smallest radii where perturbation is linear
        linear_radii = [r for r in sorted_radii if r <= 0.001]
        if not linear_radii:
            linear_radii = sorted_radii[:3] if len(sorted_radii) >= 3 else sorted_radii
            
        for current_metric in metric_list:
            # Extract layer array from the first radius
            first_radius = sorted_radii[0]
            if current_metric not in radii_data[first_radius]:
                continue
            layer_arr = np.array(radii_data[first_radius]["layers"])
            num_layers = len(layer_arr)
            
            # Helper to compute gain array for a given key in radii_data
            def compute_gain_for_key(metric_name):
                A_arr = np.zeros(num_layers)
                for i in range(num_layers):
                    ratios = []
                    for r in linear_radii:
                        if metric_name in radii_data[r] and len(radii_data[r][metric_name]) > i:
                            ratios.append(radii_data[r][metric_name][i] / r)
                    A_arr[i] = np.mean(ratios) if ratios else 1.0
                
                # Prepend 1.0 for the virtual input layer
                A_with_input = np.insert(A_arr, 0, 1.0)
                
                layer_jac_arr = np.zeros(num_layers)
                for i in range(num_layers):
                    layer_jac_arr[i] = A_with_input[i+1] / np.maximum(A_with_input[i], 1e-12)
                return A_with_input, layer_jac_arr
            
            # Compute main aggregated metric
            A_with_input, layer_jac = compute_gain_for_key(current_metric)
            layer_arr_with_input = np.insert(layer_arr, 0, layer_arr[0] - 1)
            
            # Compute fan error bounds if requested
            has_fan = (error_bars in ["fan", "percentiles"]) and ("p10" in radii_data[first_radius])
            if has_fan:
                A_p10, layer_jac_p10 = compute_gain_for_key("p10")
                A_p25, layer_jac_p25 = compute_gain_for_key("p25")
                A_p75, layer_jac_p75 = compute_gain_for_key("p75")
                A_p90, layer_jac_p90 = compute_gain_for_key("p90")
                
            # Compute individual prompt trajectories for the swarm
            swarm_data = []
            if group_key.endswith("_aggregated") and prompt_keys:
                for p_key in prompt_keys:
                    p_radii_data = data[p_key]
                    p_sorted_radii = sorted(p_radii_data.keys())
                    p_linear_radii = [r for r in p_sorted_radii if r <= 0.001]
                    if not p_linear_radii:
                        p_linear_radii = p_sorted_radii[:3] if len(p_sorted_radii) >= 3 else p_sorted_radii
                        
                    p_A = np.zeros(num_layers)
                    for i in range(num_layers):
                        ratios = []
                        for r in p_linear_radii:
                            if current_metric in p_radii_data[r] and len(p_radii_data[r][current_metric]) > i:
                                ratios.append(p_radii_data[r][current_metric][i] / r)
                        p_A[i] = np.mean(ratios) if ratios else 1.0
                        
                    p_A_with_input = np.insert(p_A, 0, 1.0)
                    p_layer_jac = np.zeros(num_layers)
                    for i in range(num_layers):
                        p_layer_jac[i] = p_A_with_input[i+1] / np.maximum(p_A_with_input[i], 1e-12)
                    swarm_data.append((p_A_with_input, p_layer_jac))
            
            # Now plot for each combination of scale
            for x_scale in x_scales:
                for y_scale in y_scales:
                    # 1. Plot Cumulative Aggregate Jacobian
                    if plot_cum:
                        plt.figure(figsize=(8, 6))
                        # Plot swarm (individual prompts) in background
                        for p_A_in, _ in swarm_data:
                            plt.plot(layer_arr_with_input, p_A_in, color='gray', alpha=0.15, linewidth=1, marker='o', markersize=2)
                        
                        # Plot baseline
                        plt.axhline(1.0, color='gray', linestyle='--', linewidth=1.5, zorder=1)
                        
                        # Plot main line
                        plt.plot(layer_arr_with_input, A_with_input, marker='o', color='b', linewidth=2, markersize=4, zorder=3)
                        
                        # Fan shading
                        if has_fan:
                            plt.fill_between(layer_arr_with_input, np.maximum(1e-12, A_p10), A_p90, color='b', alpha=0.1, zorder=2)
                            plt.fill_between(layer_arr_with_input, np.maximum(1e-12, A_p25), A_p75, color='b', alpha=0.2, zorder=2)
                            
                        plt.xlabel("Layer")
                        plt.ylabel("Linearized Perturbation Gain")
                        plt.xscale(x_scale)
                        plt.yscale(y_scale)
                        plt.grid(True, alpha=0.3, which="both")
                        if plotting_cfg.get("show_title", False):
                            plt.title(f"Extracted Aggregate Jacobian | {group_titles[group_key]}")
                        plt.tight_layout()
                        filename = f"{info_key}_extracted_cumulative_jacobian_{current_metric}_xscale-{x_scale}_yscale-{y_scale}.png"
                        plt.savefig(os.path.join(plots_dir, filename), dpi=dpi)
                        plt.close()
                        
                    # 2. Plot Layer Jacobian
                    if plot_lay:
                        plt.figure(figsize=(8, 6))
                        # Plot swarm (individual prompts) in background
                        for _, p_lj in swarm_data:
                            plt.plot(layer_arr, p_lj, color='gray', alpha=0.15, linewidth=1, marker='s', markersize=2)
                            
                        # Plot baseline
                        plt.axhline(1.0, color='gray', linestyle='--', linewidth=1.5, zorder=1)
                        
                        # Plot main line
                        plt.plot(layer_arr, layer_jac, marker='s', color='r', linewidth=2, markersize=4, zorder=3)
                        
                        # Fan shading
                        if has_fan:
                            plt.fill_between(layer_arr, np.maximum(1e-12, layer_jac_p10), layer_jac_p90, color='r', alpha=0.1, zorder=2)
                            plt.fill_between(layer_arr, np.maximum(1e-12, layer_jac_p25), layer_jac_p75, color='r', alpha=0.2, zorder=2)
                            
                        plt.xlabel("Layer")
                        plt.ylabel("Linearized Perturbation Gain")
                        plt.xscale(x_scale)
                        plt.yscale(y_scale)
                        plt.grid(True, alpha=0.3, which="both")
                        if plotting_cfg.get("show_title", False):
                            plt.title(f"Extracted Layer Jacobian | {group_titles[group_key]}")
                        plt.tight_layout()
                        filename = f"{info_key}_extracted_layer_jacobian_{current_metric}_xscale-{x_scale}_yscale-{y_scale}.png"
                        plt.savefig(os.path.join(plots_dir, filename), dpi=dpi)
                        plt.close()
                        
                    # 3. Plot both together
                    if plot_tog:
                        plt.figure(figsize=(8, 6))
                        # Plot swarm in background
                        for p_A_in, p_lj in swarm_data:
                            plt.plot(layer_arr_with_input, p_A_in, color='blue', alpha=0.08, linewidth=0.8)
                            plt.plot(layer_arr, p_lj, color='red', alpha=0.08, linewidth=0.8) 
                            
                        # Plot baseline
                        plt.axhline(1.0, color='gray', linestyle='--', linewidth=1.5, zorder=1, label="Baseline")
                        
                        # Plot main lines
                        plt.plot(layer_arr_with_input, A_with_input, marker='o', color='b', linewidth=2, markersize=4, label="Aggregate Jacobian", zorder=4)
                        plt.plot(layer_arr, layer_jac, marker='s', color='r', linewidth=2, markersize=4, label="Layer Jacobian", zorder=4)
                        
                        # Fan shading
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
                        if plotting_cfg.get("show_title", False):
                            plt.title(f"Extracted Jacobians | {group_titles[group_key]}")
                        plt.tight_layout()
                        filename = f"{info_key}_extracted_jacobians_together_{current_metric}_xscale-{x_scale}_yscale-{y_scale}.png"
                        plt.savefig(os.path.join(plots_dir, filename), dpi=dpi)
                        plt.close()

def main():
    config_path = "config.yaml"
    if not os.path.exists(config_path):
        print(f"Error: {config_path} not found.")
        return
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    results_dir = config.get("experiment", {}).get("results_dir", "./results_perturbations")
    plotting_cfg = config.get("plotting", {})
    
    # Load overrides from jacobian_config.yaml if it exists
    if os.path.exists("jacobian_config.yaml"):
        with open("jacobian_config.yaml", "r") as f_jac:
            jac_config = yaml.safe_load(f_jac)
            if "plotting" in jac_config:
                for k, v in jac_config["plotting"].items():
                    plotting_cfg[k] = v
    
    print(f"Starting plotting from pre-calculated data...")
    plot_perturbations(results_dir, plotting_cfg)

if __name__ == "__main__":
    main()
