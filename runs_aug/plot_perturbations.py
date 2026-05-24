import os
import re
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
                        
                        # Replace the label "Magnitude" with "Perturbation"
                        if len(combinations) > 1:
                            plt.legend(title="Perturbation", loc='upper left', bbox_to_anchor=(1, 1))
                        plt.tight_layout()
                        
                        metric_str = "-".join(current_metric_list)
                        plot_filename = f"{info_key}_metric-{metric_str}_err-{error_bars}_xscale-{x_scale}_yscale-{y_scale}.png"
                        plt.savefig(os.path.join(plots_dir, plot_filename), dpi=dpi)
                        plt.close()
        
                    # --- Option B: Plot each magnitude separately ---
                    if plot_magnitude_separated:
                        for radius in sorted_radii:
                            plt.figure(figsize=(10, 6))
                            colors = plt.cm.tab10(np.linspace(0, 1, max(len(current_metric_list), 10)))
                            
                            for i, current_metric in enumerate(current_metric_list):
                                color = colors[i % len(colors)]
                                metrics_dict = radii_data[radius]
                                
                                layer_arr = metrics_dict["layers"]
                                m_arr = metrics_dict[current_metric]
                                
                                line_label = f"{current_metric.capitalize()}"
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
                                title = f"{group_titles[group_key]} | R={radius}"
                                if separate_figure_metrics and len(current_metric_list) == 1:
                                    title += f" ({current_metric_list[0].capitalize()})"
                                plt.title(f"Divergence over Layers | {title}")
                                
                            plt.xlabel("Layer")
                            plt.ylabel("Distance")
                            plt.xscale(x_scale)
                            plt.yscale(y_scale)
                            plt.grid(True, alpha=0.3)
                            
                            if len(current_metric_list) > 1:
                                plt.legend(title="Metric", loc='upper left', bbox_to_anchor=(1, 1))
                            plt.tight_layout()
                            
                            metric_str = "-".join(current_metric_list)
                            plot_filename = f"{info_key}_R{radius}_metric-{metric_str}_err-{error_bars}_xscale-{x_scale}_yscale-{y_scale}.png"
                            plt.savefig(os.path.join(plots_dir, plot_filename), dpi=dpi)
                            plt.close()
 
    # 2. Distribution Heatmaps (One per Radius per Group)
    if plotting_cfg.get("plot_heatmap", True):
        for group_key, radii_data in data.items():
            if group_key.endswith("_aggregated") and not plot_prompt_together:
                continue
            if not group_key.endswith("_aggregated") and not plot_prompt_separated:
                continue
                
            info_key = get_safe_filename_info(group_key, group_titles)
            for radius, metrics_dict in radii_data.items():
                if "hist" not in metrics_dict:
                    continue
                    
                layers = metrics_dict["layers"]
                hists = metrics_dict["hist"]
                bins = metrics_dict["hist_bins"]
                
                all_hists = np.stack(hists, axis=1) # [bins, layers]
                all_hists = np.nan_to_num(all_hists, nan=0.0, posinf=0.0, neginf=0.0)
                
                vmax = float(all_hists.max())
                vmin = 1e-3
                if not np.isfinite(vmax) or vmax <= vmin:
                    vmax = vmin * 10.0
                
                plt.figure(figsize=(12, 8))
                y_bins = bins[0]
                extent = [layers[0], layers[-1], y_bins[0], y_bins[-1]]
                
                plt.imshow(all_hists, aspect='auto', origin='lower', extent=extent, cmap='magma', norm=LogNorm(vmin=vmin, vmax=vmax))
                plt.colorbar(label='Density')
                
                plt.plot(layers, metrics_dict["median"], color='cyan', linewidth=2, label='Median')
                plt.plot(layers, metrics_dict["p10"], color='cyan', linestyle='--', alpha=0.5, label='10th/90th P')
                plt.plot(layers, metrics_dict["p90"], color='cyan', linestyle='--', alpha=0.5)
                
                if plotting_cfg.get("show_title", False):
                    plt.title(f"Distribution Evolution | {group_titles[group_key]} | R={radius}")
                plt.xlabel("Layer")
                plt.ylabel("Distance")
                plt.legend()
                
                plot_filename = f"{info_key}_heatmap_R{radius}.png"
                plt.savefig(os.path.join(plots_dir, plot_filename), dpi=dpi)
                plt.close()
 
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
                            setup_name = group_key.replace("_aggregated", "")
                            prompt_keys = [k for k in data.keys() if k.startswith(setup_name) and not k.endswith("_aggregated")]
                            
                            if prompt_keys:
                                plt.figure(figsize=(8, 6))
                                cmap = plt.cm.rainbow(np.linspace(0, 1, len(prompt_keys)))
                                
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
                                        plt.plot(px_radii, py_avg, color=cmap[p_idx], alpha=0.3, linewidth=1.5, linestyle='-')
                                
                                # Overlay aggregated mean
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

def main():
    config_path = "config.yaml"
    if not os.path.exists(config_path):
        print(f"Error: {config_path} not found.")
        return
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    results_dir = config.get("experiment", {}).get("results_dir", "./results_perturbations")
    plotting_cfg = config.get("plotting", {})
    
    print(f"Starting plotting from pre-calculated data...")
    plot_perturbations(results_dir, plotting_cfg)

if __name__ == "__main__":
    main()
