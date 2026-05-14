import os
import yaml
import pickle
import numpy as np
import matplotlib.pyplot as plt

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
                if m_name not in group_data:
                    continue
                    
                layer_arr = group_data[m_name]["layers"]
                
                for stat_metric in current_metric_list:
                    m_arr = group_data[m_name][stat_metric]
                    e_arr = group_data[m_name][self.error_bars]
                    
                    color = cmap[color_idx % len(cmap)] if colors is None else cmap[m_idx % len(cmap)]
                    color_idx += 1
                    
                    base_label = labels[m_idx] if labels else m_name
                    line_label = f"{base_label} ({stat_metric.capitalize()})" if len(current_metric_list) > 1 else base_label
                    
                    plt.plot(layer_arr, m_arr, marker='o', color=color, label=line_label)
                    if self.error_bars != "none":
                        plt.fill_between(layer_arr, m_arr - e_arr, m_arr + e_arr, color=color, alpha=0.2)
                        
            if hlines:
                for h in hlines:
                    plt.axhline(y=h['y'], color=h.get('color', 'black'), linestyle=h.get('linestyle', '--'), label=h.get('label', ''))
                    
            title = f"{title_prefix} | {self.group_titles[group_key]}"
            if self.separate_figure_metrics and len(current_metric_list) == 1:
                title += f" ({current_metric_list[0].capitalize()})"
                
            plt.title(title, fontsize=14)
            plt.xlabel("Layer Index", fontsize=12)
            
            y_ax_label = ylabel
            if len(current_metric_list) == 1:
                y_ax_label = f"{current_metric_list[0].capitalize()} " + ylabel
                if self.error_bars != "none":
                    y_ax_label += f" (± {self.error_bars})"
                    
            plt.ylabel(y_ax_label, fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.legend(title="Metric")
            plt.tight_layout()
            
            safe_key = group_key.replace(" ", "_").replace("/", "-")
            metric_str = "-".join(current_metric_list)
            plot_filename = f"{filename_prefix}_{safe_key}_metric-{metric_str}_err-{self.error_bars}.png"
            plot_path = os.path.join(self.plots_dir, plot_filename)
            plt.savefig(plot_path, dpi=300)
            print(f"Generated plot: {plot_path}")
            plt.close()

    def plot_all(self):
        print("--- Generating Jacobian Plots ---")
        for group_key in self.data.keys():
            if not self._should_plot(group_key):
                continue
                
            # 1. Spectral Norms
            self._plot_metric_across_layers(
                group_key, 
                ["spectral_norms"], 
                r"Local Jacobian Spectral Norm $\|J_{MLP}\|_2$", 
                r"Spectral Norm $\|J\|_2$", 
                "spectral_norms",
                labels=["Spectral Norm"],
                colors=['darkred'],
                hlines=[{'y': 1.0, 'label': 'Neutral Boundary (||J||_2 = 1)'}]
            )
            
            # 2. Lambda True
            self._plot_metric_across_layers(
                group_key, 
                ["lambda_true"], 
                r"Mean Squared Singular Value $\bar{\lambda}_{true}$", 
                r"$\bar{\lambda}_{true} = \frac{1}{d} \|J\|_F^2$", 
                "lambda_true",
                labels=[r"$\bar{\lambda}_{true}$"],
                colors=['navy'],
                hlines=[{'y': 1.0, 'label': 'Neutral Boundary'}]
            )
            
            # 3. Weight SVDs (Scalar metrics, we still use the same logic but error_bars will naturally be 0 if single prompt)
            self._plot_metric_across_layers(
                group_key, 
                ["W_gate_max_SVD", "W_up_max_SVD", "W_down_max_SVD"], 
                "Maximum Singular Values of SwiGLU Weight Matrices", 
                "Max Singular Value", 
                "weight_svds",
                labels=[r'$W_{gate}$ Max SVD', r'$W_{up}$ Max SVD', r'$W_{down}$ Max SVD'],
                hlines=[{'y': 1.0, 'label': 'Neutral Boundary'}]
            )
            
            # 4. Scaled Frobenius
            self._plot_metric_across_layers(
                group_key, 
                ["W_gate_scaled_F2", "W_up_scaled_F2", "W_down_scaled_F2"], 
                "Scaled Frobenius Traces of Weight Matrices", 
                "Scaled Squared Frobenius Norm", 
                "scaled_frobenius",
                labels=[r'$W_{gate}$ Scaled $\| \cdot \|_F^2$', r'$W_{up}$ Scaled $\| \cdot \|_F^2$', r'$W_{down}$ Scaled $\| \cdot \|_F^2$'],
                hlines=[{'y': 1.0, 'label': 'Neutral Boundary'}]
            )
            
            # 5. Activation Densities
            self._plot_metric_across_layers(
                group_key, 
                ["S_x_sq_mean", "D_x_sq_mean"], 
                "Activation Density / Magnitude Across Layers", 
                "Squared Magnitude", 
                "activation_densities",
                labels=[r'$S(x)^2$', r'$D(x)^2$'],
                colors=['teal', 'orange']
            )
        print("-" * 50)

def main():
    config_path = "jacobian_config.yaml"
    if not os.path.exists(config_path):
        print(f"Error: {config_path} not found.")
        return
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    results_dir = config.get("experiment", {}).get("results_dir", "./results_jacobians")
    plotting_cfg = config.get("plotting", {})
    
    data_path = os.path.join(results_dir, "analyzed_jacobians.pkl")
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Run analyze_jacobians.py first.")
        return
        
    print(f"Starting plotting from pre-calculated data...")
    plotter = JacobianPlotter(data_path, plotting_cfg)
    plotter.plot_all()

if __name__ == "__main__":
    main()
