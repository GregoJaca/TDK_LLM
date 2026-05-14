import os
import yaml
import pickle
import numpy as np
import matplotlib.pyplot as plt

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
    
    # 1. Standard Divergence Plots
    metrics_to_process = [[m] for m in metric_list] if separate_figure_metrics else [metric_list]
    
    for current_metric_list in metrics_to_process:
        for group_key, radii_data in data.items():
            
            # Filter groups based on flags
            if group_key.endswith("_aggregated") and not plot_prompt_together:
                continue
            if not group_key.endswith("_aggregated") and not plot_prompt_separated:
                continue
                
            plt.figure(figsize=(10, 6))
            sorted_radii = sorted(radii_data.keys())
            
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
                e_arr = metrics_dict[error_bars]
                
                line_label = f"R={radius} | {current_metric.capitalize()}" if len(current_metric_list) > 1 else f"{radius}"
                
                plt.plot(layer_arr, m_arr, marker='o', color=color, label=line_label)
                if error_bars != "none":
                    plt.fill_between(layer_arr, m_arr - e_arr, m_arr + e_arr, color=color, alpha=0.2)
                    
            title = group_titles[group_key]
            if separate_figure_metrics and len(current_metric_list) == 1:
                title += f" ({current_metric_list[0].capitalize()})"
                
            plt.title(f"Divergence over Layers | {title}", fontsize=14)
            plt.xlabel("Layer Index", fontsize=12)
            
            ylabel = "Distance"
            if len(current_metric_list) == 1:
                ylabel = f"{current_metric_list[0].capitalize()} " + ylabel
                if error_bars != "none":
                    ylabel += f" (± {error_bars})"
            else:
                ylabel = "Aggregated Distances"
                
            plt.ylabel(ylabel, fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.legend(title="Magnitude")
            plt.tight_layout()
            
            safe_key = group_key.replace(" ", "_").replace("/", "-")
            metric_str = "-".join(current_metric_list)
            plot_filename = f"{safe_key}_metric-{metric_str}_err-{error_bars}.png"
            plot_path = os.path.join(plots_dir, plot_filename)
            plt.savefig(plot_path, dpi=300)
            print(f"Generated plot: {plot_path}")
            plt.close()

    # 2. Average over Layers 2-25 vs Perturbation Magnitude Plot
    for current_metric in metric_list:
        for group_key, radii_data in data.items():
            if group_key.endswith("_aggregated") and not plot_prompt_together:
                continue
            if not group_key.endswith("_aggregated") and not plot_prompt_separated:
                continue

            sorted_radii = sorted(radii_data.keys())
            x_radii = []
            y_avg = []
            y_err = []

            for radius in sorted_radii:
                metrics_dict = radii_data[radius]
                layer_arr = metrics_dict["layers"]
                m_arr = metrics_dict[current_metric]
                
                # Find indices for layers between 2 and 25 inclusive
                mask = (layer_arr >= 2) & (layer_arr <= 25)
                if not np.any(mask):
                    continue
                    
                selected_vals = m_arr[mask]
                
                x_radii.append(radius)
                y_avg.append(np.mean(selected_vals))
                
                if error_bars == "std":
                    # The overall variance across the layers is the mean of the layer-wise variances 
                    # plus the variance of the layer-wise means (Law of Total Variance).
                    var_arr = metrics_dict["var"]
                    selected_vars = var_arr[mask]
                    overall_var = np.mean(selected_vars) + np.var(selected_vals)
                    y_err.append(np.sqrt(overall_var))
                elif error_bars == "var":
                    var_arr = metrics_dict["var"]
                    selected_vars = var_arr[mask]
                    overall_var = np.mean(selected_vars) + np.var(selected_vals)
                    y_err.append(overall_var)
                else:
                    y_err.append(0)

            if not x_radii:
                continue

            plt.figure(figsize=(8, 6))
            
            # Matplotlib's errorbar doesn't always show up well on log-log if the bottom error goes below 0.
            # We calculate asymmetrical error bars for log scale to prevent negative values.
            if error_bars != "none":
                y_err_np = np.array(y_err)
                y_avg_np = np.array(y_avg)
                
                # Prevent lower error bar from going to <= 0 in log scale
                lower_error = np.clip(y_err_np, 0, y_avg_np * 0.999)
                upper_error = y_err_np
                
                plt.errorbar(x_radii, y_avg_np, yerr=[lower_error, upper_error], marker='o', capsize=5, linestyle='-', color='b')
            else:
                plt.plot(x_radii, y_avg, marker='o', linestyle='-', color='b')

            plt.xscale('log') 
            plt.yscale('log') 
            
            title = group_titles[group_key]
            plt.title(f"Avg Divergence (Layers 2-25) vs Radius | {title}", fontsize=12)
            plt.xlabel("Perturbation Magnitude (Radius)", fontsize=12)
            
            ylabel = f"Average {current_metric.capitalize()} Distance"
            if error_bars != "none":
                ylabel += f" (± std dev across layers)"
            plt.ylabel(ylabel, fontsize=12)
            
            plt.grid(True, alpha=0.3, which="both")
            plt.tight_layout()

            safe_key = group_key.replace(" ", "_").replace("/", "-")
            plot_filename = f"{safe_key}_avgL2-25_{current_metric}.png"
            plot_path = os.path.join(plots_dir, plot_filename)
            plt.savefig(plot_path, dpi=300)
            print(f"Generated avg plot: {plot_path}")
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
