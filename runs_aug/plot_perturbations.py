import os
import yaml
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

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
    
    metric_list = plotting_cfg.get("metric", ["mean"])
    if isinstance(metric_list, str):
        metric_list = [metric_list]
        
    error_bars = plotting_cfg.get("error_bars", "none")
    plot_prompt_together = plotting_cfg.get("plot_prompt_together", False)
    plot_prompt_separated = plotting_cfg.get("plot_prompt_separated", True)
    separate_figure_metrics = plotting_cfg.get("separate_figure_metrics", False)
    plot_magnitude_together = plotting_cfg.get("plot_magnitude_together", True)
    plot_magnitude_separated = plotting_cfg.get("plot_magnitude_separated", True)
    
    # 1. Standard Divergence Plots with Fan Shading
    metrics_to_process = [[m] for m in metric_list] if separate_figure_metrics else [metric_list]
    
    for current_metric_list in metrics_to_process:
        for group_key, radii_data in data.items():
            if group_key.endswith("_aggregated") and not plot_prompt_together:
                continue
            if not group_key.endswith("_aggregated") and not plot_prompt_separated:
                continue
                
            sorted_radii = sorted(radii_data.keys())
            
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
                            plt.fill_between(layer_arr, metrics_dict["p10"], metrics_dict["p90"], color=color, alpha=0.1, label='_nolegend_')
                        if "p25" in metrics_dict and "p75" in metrics_dict:
                            plt.fill_between(layer_arr, metrics_dict["p25"], metrics_dict["p75"], color=color, alpha=0.2, label='_nolegend_')
                    elif error_bars in metrics_dict:
                        e_arr = metrics_dict[error_bars]
                        # Handle std/var clipping for positive metrics
                        lower = np.maximum(0, m_arr - e_arr) if error_bars in ["std", "var"] else m_arr - e_arr
                        plt.fill_between(layer_arr, lower, m_arr + e_arr, color=color, alpha=0.2, label='_nolegend_')
                        
                title = group_titles[group_key]
                if separate_figure_metrics and len(current_metric_list) == 1:
                    title += f" ({current_metric_list[0].capitalize()})"
                    
                plt.title(f"Divergence over Layers | {title}", fontsize=14)
                plt.xlabel("Layer Index", fontsize=12)
                plt.ylabel("Distance", fontsize=12)
                plt.grid(True, alpha=0.3)
                plt.legend(title="Magnitude", loc='upper left', bbox_to_anchor=(1, 1))
                plt.tight_layout()
                
                safe_key = group_key.replace(" ", "_").replace("/", "-")
                metric_str = "-".join(current_metric_list)
                plot_filename = f"{safe_key}_metric-{metric_str}_err-{error_bars}.png"
                plt.savefig(os.path.join(plots_dir, plot_filename), dpi=300)
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
                                plt.fill_between(layer_arr, metrics_dict["p10"], metrics_dict["p90"], color=color, alpha=0.1, label='_nolegend_')
                            if "p25" in metrics_dict and "p75" in metrics_dict:
                                plt.fill_between(layer_arr, metrics_dict["p25"], metrics_dict["p75"], color=color, alpha=0.2, label='_nolegend_')
                        elif error_bars in metrics_dict:
                            e_arr = metrics_dict[error_bars]
                            # Handle std/var clipping for positive metrics
                            lower = np.maximum(0, m_arr - e_arr) if error_bars in ["std", "var"] else m_arr - e_arr
                            plt.fill_between(layer_arr, lower, m_arr + e_arr, color=color, alpha=0.2, label='_nolegend_')
                            
                    title = f"{group_titles[group_key]} | R={radius}"
                    if separate_figure_metrics and len(current_metric_list) == 1:
                        title += f" ({current_metric_list[0].capitalize()})"
                        
                    plt.title(f"Divergence over Layers | {title}", fontsize=14)
                    plt.xlabel("Layer Index", fontsize=12)
                    plt.ylabel("Distance", fontsize=12)
                    plt.grid(True, alpha=0.3)
                    if len(current_metric_list) > 1:
                        plt.legend(title="Metric", loc='upper left', bbox_to_anchor=(1, 1))
                    plt.tight_layout()
                    
                    safe_key = group_key.replace(" ", "_").replace("/", "-")
                    metric_str = "-".join(current_metric_list)
                    plot_filename = f"{safe_key}_R{radius}_metric-{metric_str}_err-{error_bars}.png"
                    plt.savefig(os.path.join(plots_dir, plot_filename), dpi=300)
                    plt.close()

    # 2. Distribution Heatmaps (One per Radius per Group)
    for group_key, radii_data in data.items():
        if group_key.endswith("_aggregated") and not plot_prompt_together:
            continue
        if not group_key.endswith("_aggregated") and not plot_prompt_separated:
            continue
            
        for radius, metrics_dict in radii_data.items():
            if "hist" not in metrics_dict:
                continue
                
            layers = metrics_dict["layers"]
            hists = metrics_dict["hist"]
            bins = metrics_dict["hist_bins"]
            
            # Combine histograms into a 2D array
            # Assuming all bins are the same or we need to interpolate
            all_hists = np.stack(hists, axis=1) # [bins, layers]
            
            plt.figure(figsize=(12, 8))
            # Use the first bin edges for Y axis
            y_bins = bins[0]
            extent = [layers[0], layers[-1], y_bins[0], y_bins[-1]]
            
            plt.imshow(all_hists, aspect='auto', origin='lower', extent=extent, cmap='magma', norm=LogNorm(vmin=1e-3, vmax=all_hists.max()))
            plt.colorbar(label='Density')
            
            # Overlay median line
            plt.plot(layers, metrics_dict["median"], color='cyan', linewidth=2, label='Median')
            plt.plot(layers, metrics_dict["p10"], color='cyan', linestyle='--', alpha=0.5, label='10th/90th P')
            plt.plot(layers, metrics_dict["p90"], color='cyan', linestyle='--', alpha=0.5)
            
            plt.title(f"Distribution Evolution | {group_titles[group_key]} | R={radius}", fontsize=14)
            plt.xlabel("Layer Index", fontsize=12)
            plt.ylabel("Distance Value", fontsize=12)
            plt.legend()
            
            safe_key = group_key.replace(" ", "_").replace("/", "-")
            plot_filename = f"{safe_key}_heatmap_R{radius}.png"
            plt.savefig(os.path.join(plots_dir, plot_filename), dpi=300)
            plt.close()

    # 3. Average over Layers 2-25 vs Perturbation Magnitude Plot (unchanged but using robust error)
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
                
                if error_bars in ["std", "var"]:
                    var_arr = metrics_dict["var"]
                    overall_var = np.mean(var_arr[mask]) + np.var(selected_vals)
                    err = np.sqrt(overall_var) if error_bars == "std" else overall_var
                    y_err_low.append(err)
                    y_err_up.append(err)
                elif error_bars == "fan" or error_bars == "percentiles":
                    # Use p10-p90 spread as error bounds
                    low_spread = np.mean(metrics_dict["p10"][mask])
                    high_spread = np.mean(metrics_dict["p90"][mask])
                    y_err_low.append(np.mean(selected_vals) - low_spread)
                    y_err_up.append(high_spread - np.mean(selected_vals))
                else:
                    y_err_low.append(0); y_err_up.append(0)

            if not x_radii: continue

            plt.figure(figsize=(8, 6))
            if any(y_err_up):
                # Prevent log(negative)
                low_err = np.clip(y_err_low, 0, np.array(y_avg) * 0.99)
                plt.errorbar(x_radii, y_avg, yerr=[low_err, y_err_up], marker='o', capsize=5, color='b')
            else:
                plt.plot(x_radii, y_avg, marker='o', color='b')

            plt.xscale('log'); plt.yscale('log')
            plt.title(f"Scaling Law (L2-25) | {group_titles[group_key]}", fontsize=12)
            plt.xlabel("Radius", fontsize=12); plt.ylabel(f"Avg {current_metric.capitalize()}", fontsize=12)
            plt.grid(True, alpha=0.3, which="both")
            plt.tight_layout()
            safe_key = group_key.replace(" ", "_").replace("/", "-")
            plt.savefig(os.path.join(plots_dir, f"{safe_key}_scaling_{current_metric}.png"), dpi=300)
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
