import os
import yaml
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

class JacobianPlotter:
    def __init__(self, analyzed_data_path, plotting_cfg):
        self.data_path = analyzed_data_path
        self.plotting_cfg = plotting_cfg
        
        with open(self.data_path, "rb") as f:
            self.analyzed_data = pickle.load(f)
            
        self.group_titles = self.analyzed_data["group_titles"]
        self.data = self.analyzed_data["data"]
        
        base_dir = os.path.dirname(analyzed_data_path)
        self.plots_dir = os.path.join(base_dir, "aggregated_plots")
        os.makedirs(self.plots_dir, exist_ok=True)
        
        self.metric_list = self.plotting_cfg.get("metric", ["mean"])
        if isinstance(self.metric_list, str):
            self.metric_list = [self.metric_list]
            
        self.error_bars = self.plotting_cfg.get("error_bars", "none")
        self.plot_prompt_together = self.plotting_cfg.get("plot_prompt_together", False)
        self.plot_prompt_separated = self.plotting_cfg.get("plot_prompt_separated", True)
        self.separate_figure_metrics = self.plotting_cfg.get("separate_figure_metrics", False)
        
    def _should_plot(self, group_key):
        if group_key.endswith("_aggregated") and not self.plot_prompt_together:
            return False
        if not group_key.endswith("_aggregated") and not self.plot_prompt_separated:
            return False
        return True
        
    def _plot_metric_across_layers(self, group_key, metric_names, title_prefix, ylabel, filename_prefix, labels=None, colors=None, hlines=None):
        metrics_to_process = [[m] for m in self.metric_list] if self.separate_figure_metrics else [self.metric_list]
        group_data = self.data[group_key]
        
        for current_metric_list in metrics_to_process:
            plt.figure(figsize=(10, 6))
            if colors is None:
                cmap = plt.cm.tab10(np.linspace(0, 1, max(len(metric_names) * len(current_metric_list), 10)))
            else:
                cmap = colors
                
            color_idx = 0
            for m_idx, m_name in enumerate(metric_names):
                if m_name not in group_data: continue
                layer_arr = group_data[m_name]["layers"]
                
                for stat_metric in current_metric_list:
                    m_arr = group_data[m_name][stat_metric]
                    color = cmap[color_idx % len(cmap)] if colors is None else cmap[m_idx % len(cmap)]
                    color_idx += 1
                    
                    base_label = labels[m_idx] if labels else m_name
                    line_label = f"{base_label} ({stat_metric.capitalize()})" if len(current_metric_list) > 1 else base_label
                    plt.plot(layer_arr, m_arr, marker='o', color=color, label=line_label, linewidth=2, markersize=4)
                    
                    # Fan Shading / Error Bars
                    if self.error_bars == "fan" or self.error_bars == "percentiles":
                        if "p10" in group_data[m_name] and "p90" in group_data[m_name]:
                            plt.fill_between(layer_arr, group_data[m_name]["p10"], group_data[m_name]["p90"], color=color, alpha=0.1)
                        if "p25" in group_data[m_name] and "p75" in group_data[m_name]:
                            plt.fill_between(layer_arr, group_data[m_name]["p25"], group_data[m_name]["p75"], color=color, alpha=0.2)
                    elif self.error_bars in group_data[m_name]:
                        e_arr = group_data[m_name][self.error_bars]
                        lower = np.maximum(0, m_arr - e_arr) if self.error_bars in ["std", "var"] else m_arr - e_arr
                        plt.fill_between(layer_arr, lower, m_arr + e_arr, color=color, alpha=0.2)
            
            if hlines:
                for h in hlines:
                    plt.axhline(y=h['y'], color=h.get('color', 'black'), linestyle=h.get('linestyle', '--'), label=h.get('label', ''))
                    
            title = f"{title_prefix} | {self.group_titles[group_key]}"
            plt.title(title, fontsize=14); plt.xlabel("Layer Index", fontsize=12); plt.ylabel(ylabel, fontsize=12)
            plt.grid(True, alpha=0.3); plt.legend(title="Metric", loc='upper left', bbox_to_anchor=(1, 1))
            plt.tight_layout()
            
            safe_key = group_key.replace(" ", "_").replace("/", "-")
            metric_str = "-".join(current_metric_list)
            plt.savefig(os.path.join(self.plots_dir, f"{filename_prefix}_{safe_key}_{metric_str}.png"), dpi=300)
            plt.close()

    def plot_distribution_heatmaps(self):
        print("Generating Jacobian Distribution Heatmaps...")
        # Only for token metrics (which have distributions)
        token_metrics = ["spectral_norms", "lambda_true"]
        for group_key in self.data.keys():
            if not self._should_plot(group_key): continue
            group_data = self.data[group_key]
            
            for m_name in token_metrics:
                if m_name not in group_data or "hist" not in group_data[m_name]: continue
                
                metrics_dict = group_data[m_name]
                layers = metrics_dict["layers"]
                all_hists = np.stack(metrics_dict["hist"], axis=1)
                all_hists = np.nan_to_num(all_hists, nan=0.0, posinf=0.0, neginf=0.0)
                y_bins = metrics_dict["hist_bins"][0]
                
                vmax = float(all_hists.max())
                vmin = 1e-3
                if not np.isfinite(vmax) or vmax <= vmin:
                    vmax = vmin * 10.0
                
                plt.figure(figsize=(12, 8))
                extent = [layers[0], layers[-1], y_bins[0], y_bins[-1]]
                plt.imshow(all_hists, aspect='auto', origin='lower', extent=extent, cmap='magma', norm=LogNorm(vmin=vmin, vmax=vmax))
                plt.colorbar(label='Density')
                
                # Overlay
                plt.plot(layers, metrics_dict["median"], color='cyan', linewidth=2, label='Median')
                plt.plot(layers, metrics_dict["p10"], color='cyan', linestyle='--', alpha=0.5, label='10th/90th P')
                plt.plot(layers, metrics_dict["p90"], color='cyan', linestyle='--', alpha=0.5)
                
                plt.title(f"Jacobian Distribution: {m_name} | {self.group_titles[group_key]}", fontsize=14)
                plt.xlabel("Layer Index", fontsize=12); plt.ylabel(f"{m_name} Value", fontsize=12)
                plt.legend(); plt.tight_layout()
                
                safe_key = group_key.replace(" ", "_").replace("/", "-")
                plt.savefig(os.path.join(self.plots_dir, f"heatmap_{m_name}_{safe_key}.png"), dpi=300)
                plt.close()

    def plot_all(self):
        print("--- Generating Jacobian Plots ---")
        for group_key in self.data.keys():
            if not self._should_plot(group_key): continue
            self._plot_metric_across_layers(group_key, ["spectral_norms"], r"Jacobian Spectral Norm $\|J_{MLP}\|_2$", r"$\|J\|_2$", "spectral_norms", labels=["Spectral Norm"], colors=['darkred'], hlines=[{'y': 1.0, 'label': 'Neutral Boundary'}])
            self._plot_metric_across_layers(group_key, ["lambda_true"], r"Mean Squared Singular Value $\bar{\lambda}_{true}$", r"$\bar{\lambda}_{true}$", "lambda_true", labels=[r"$\bar{\lambda}_{true}$"], colors=['navy'], hlines=[{'y': 1.0, 'label': 'Neutral Boundary'}])
            self._plot_metric_across_layers(group_key, ["W_gate_max_SVD", "W_up_max_SVD", "W_down_max_SVD"], "Weight Matrix Spectral Norms", "Max Singular Value", "weight_svds", labels=[r'$W_{gate}$', r'$W_{up}$', r'$W_{down}$'], hlines=[{'y': 1.0, 'label': 'Neutral Boundary'}])
            self._plot_metric_across_layers(group_key, ["W_gate_scaled_F2", "W_up_scaled_F2", "W_down_scaled_F2"], "Scaled Frobenius Norms", "Scaled $\| \cdot \|_F^2$", "scaled_frobenius", labels=[r'$W_{gate}$', r'$W_{up}$', r'$W_{down}$'], hlines=[{'y': 1.0, 'label': 'Neutral Boundary'}])
            self._plot_metric_across_layers(group_key, ["S_x_sq_mean", "D_x_sq_mean"], "Activation Densities", "Squared Magnitude", "activation_densities", labels=[r'$S(x)^2$', r'$D(x)^2$'], colors=['teal', 'orange'])
        
        self.plot_distribution_heatmaps()
        print("-" * 50)

def main():
    config_path = "jacobian_config.yaml"
    if not os.path.exists(config_path): return
    with open(config_path, "r") as f: config = yaml.safe_load(f)
    results_dir = config.get("experiment", {}).get("results_dir", "./results_jacobians")
    plotting_cfg = config.get("plotting", {})
    data_path = os.path.join(results_dir, "analyzed_jacobians.pkl")
    if not os.path.exists(data_path): return
    plotter = JacobianPlotter(data_path, plotting_cfg)
    plotter.plot_all()

if __name__ == "__main__":
    main()
