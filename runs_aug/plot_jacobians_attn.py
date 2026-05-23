import os
import yaml
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

class AttentionJacobianPlotter:
    def __init__(self, analyzed_data_path, plotting_cfg):
        self.data_path = analyzed_data_path
        self.plotting_cfg = plotting_cfg
        
        with open(self.data_path, "rb") as f:
            self.analyzed_data = pickle.load(f)
            
        self.group_titles = self.analyzed_data["group_titles"]
        self.data = self.analyzed_data["data"]
        self.found_N_list = self.analyzed_data["found_N_list"]
        
        base_dir = os.path.dirname(analyzed_data_path)
        self.plots_dir = os.path.join(base_dir, "aggregated_plots_attn")
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
        
    def _get_color_for_N(self, n):
        # Standards: 20 -> Blue, 100 -> Orange, 1000 -> Green
        if n == 20:
            return "#1f77b4" # Blue
        elif n == 100:
            return "#ff7f0e" # Orange
        elif n == 1000:
            return "#2ca02c" # Green
        else:
            # Fallback to standard colormap for other values
            return plt.cm.tab10(self.found_N_list.index(n) % 10)
            
    def plot_spectral_norms(self, group_key):
        """Plot 1: Attention Global Spectral Norm"""
        group_data = self.data[group_key]
        x_scales = self.plotting_cfg.get("x_scales", ["linear"])
        y_scales = self.plotting_cfg.get("y_scales", ["linear"])
        
        metrics_to_process = [[m] for m in self.metric_list] if self.separate_figure_metrics else [self.metric_list]
        
        for x_scale in x_scales:
            for y_scale in y_scales:
                for stat_metrics in metrics_to_process:
                    plt.figure(figsize=(10, 6))
                    
                    for n in self.found_N_list:
                        m_name = f"attn_spectral_norm_N-{n}"
                        if m_name not in group_data:
                            continue
                            
                        layer_arr = group_data[m_name]["layers"]
                        color = self._get_color_for_N(n)
                        
                        for stat_metric in stat_metrics:
                            m_arr = group_data[m_name][stat_metric]
                            line_label = f"N = {n}" if len(stat_metrics) == 1 else f"N = {n} ({stat_metric.capitalize()})"
                            plt.plot(layer_arr, m_arr, marker='o', color=color, label=line_label, linewidth=2, markersize=4)
                            
                            # Error Shading
                            if self.error_bars == "fan" or self.error_bars == "percentiles":
                                if "p10" in group_data[m_name] and "p90" in group_data[m_name]:
                                    plt.fill_between(layer_arr, group_data[m_name]["p10"], group_data[m_name]["p90"], color=color, alpha=0.1)
                                if "p25" in group_data[m_name] and "p75" in group_data[m_name]:
                                    plt.fill_between(layer_arr, group_data[m_name]["p25"], group_data[m_name]["p75"], color=color, alpha=0.2)
                            else:
                                err_key = f"{stat_metric}_{self.error_bars}"
                                if err_key in group_data[m_name]:
                                    e_arr = group_data[m_name][err_key]
                                elif self.error_bars in group_data[m_name]:
                                    e_arr = group_data[m_name][self.error_bars]
                                else:
                                    e_arr = None
                                    
                                if e_arr is not None:
                                    lower = np.maximum(0, m_arr - e_arr)
                                    if y_scale == "log":
                                        lower = np.maximum(1e-12, lower)
                                    plt.fill_between(layer_arr, lower, m_arr + e_arr, color=color, alpha=0.2)
                                    
                    # Critical Baseline y = 1.0
                    plt.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, label='Chaotic Boundary (y=1.0)')
                    
                    plt.title(r"Attention Jacobian Global Spectral Norm $\|J_{attn}\|_2$" + f"\n{self.group_titles[group_key]}", fontsize=12)
                    plt.xlabel("Layer Index", fontsize=11)
                    plt.ylabel(r"$\|J_{attn}\|_2$", fontsize=11)
                    plt.xscale(x_scale)
                    plt.yscale(y_scale)
                    plt.grid(True, alpha=0.3)
                    plt.legend(loc='best')
                    plt.tight_layout()
                    
                    safe_key = group_key.replace(" ", "_").replace("/", "-")
                    metric_str = "-".join(stat_metrics)
                    filename = f"attn_spectral_norms_{safe_key}_{metric_str}_xscale-{x_scale}_yscale-{y_scale}.png"
                    plt.savefig(os.path.join(self.plots_dir, filename), dpi=300)
                    plt.close()

    def plot_attention_entropy(self, group_key):
        """Plot 2: Dynamic Attention Entropy"""
        group_data = self.data[group_key]
        x_scales = self.plotting_cfg.get("x_scales", ["linear"])
        y_scales = self.plotting_cfg.get("y_scales", ["linear"])
        
        metrics_to_process = [[m] for m in self.metric_list] if self.separate_figure_metrics else [self.metric_list]
        
        for x_scale in x_scales:
            for y_scale in y_scales:
                for stat_metrics in metrics_to_process:
                    plt.figure(figsize=(10, 6))
                    
                    for n in self.found_N_list:
                        m_name = f"mean_attn_entropy_N-{n}"
                        if m_name not in group_data:
                            continue
                            
                        layer_arr = group_data[m_name]["layers"]
                        color = self._get_color_for_N(n)
                        
                        for stat_metric in stat_metrics:
                            m_arr = group_data[m_name][stat_metric]
                            line_label = f"N = {n}" if len(stat_metrics) == 1 else f"N = {n} ({stat_metric.capitalize()})"
                            plt.plot(layer_arr, m_arr, marker='o', color=color, label=line_label, linewidth=2, markersize=4)
                            
                            # Error Shading
                            if self.error_bars == "fan" or self.error_bars == "percentiles":
                                if "p10" in group_data[m_name] and "p90" in group_data[m_name]:
                                    plt.fill_between(layer_arr, group_data[m_name]["p10"], group_data[m_name]["p90"], color=color, alpha=0.1)
                                if "p25" in group_data[m_name] and "p75" in group_data[m_name]:
                                    plt.fill_between(layer_arr, group_data[m_name]["p25"], group_data[m_name]["p75"], color=color, alpha=0.2)
                            else:
                                err_key = f"{stat_metric}_{self.error_bars}"
                                if err_key in group_data[m_name]:
                                    e_arr = group_data[m_name][err_key]
                                elif self.error_bars in group_data[m_name]:
                                    e_arr = group_data[m_name][self.error_bars]
                                else:
                                    e_arr = None
                                    
                                if e_arr is not None:
                                    lower = np.maximum(0, m_arr - e_arr)
                                    plt.fill_between(layer_arr, lower, m_arr + e_arr, color=color, alpha=0.2)
                                    
                        # Draw theoretical max entropy line for this N
                        max_ent = np.log2(n / 2.0)
                        plt.axhline(y=max_ent, color=color, linestyle='--', alpha=0.5, label=f"Max H (N={n}): {max_ent:.2f}")
                        
                    plt.title(f"Dynamic Attention Entropy (Shannon)\n{self.group_titles[group_key]}", fontsize=12)
                    plt.xlabel("Layer Index", fontsize=11)
                    plt.ylabel("Entropy (bits)", fontsize=11)
                    plt.xscale(x_scale)
                    plt.yscale(y_scale)
                    plt.grid(True, alpha=0.3)
                    plt.legend(loc='best')
                    plt.tight_layout()
                    
                    safe_key = group_key.replace(" ", "_").replace("/", "-")
                    metric_str = "-".join(stat_metrics)
                    filename = f"attn_entropy_{safe_key}_{metric_str}_xscale-{x_scale}_yscale-{y_scale}.png"
                    plt.savefig(os.path.join(self.plots_dir, filename), dpi=300)
                    plt.close()

    def plot_static_weights(self, group_key):
        """Plot 3: Static Weight Amplifiers"""
        group_data = self.data[group_key]
        x_scales = self.plotting_cfg.get("x_scales", ["linear"])
        y_scales = self.plotting_cfg.get("y_scales", ["linear"])
        
        metrics_to_process = [[m] for m in self.metric_list] if self.separate_figure_metrics else [self.metric_list]
        
        for x_scale in x_scales:
            for y_scale in y_scales:
                for stat_metrics in metrics_to_process:
                    plt.figure(figsize=(10, 6))
                    
                    # Routing norm
                    m_routing = "routing_weight_norm"
                    if m_routing in group_data:
                        layer_arr = group_data[m_routing]["layers"]
                        color_r = "crimson"
                        for stat_metric in stat_metrics:
                            m_arr = group_data[m_routing][stat_metric]
                            lbl = r"Routing $\|W_Q W_K^T\|_2$" if len(stat_metrics) == 1 else f"Routing ({stat_metric.capitalize()})"
                            plt.plot(layer_arr, m_arr, marker='s', color=color_r, label=lbl, linewidth=2, markersize=4)
                            
                            # Error
                            if self.error_bars == "fan" or self.error_bars == "percentiles":
                                if "p10" in group_data[m_routing] and "p90" in group_data[m_routing]:
                                    plt.fill_between(layer_arr, group_data[m_routing]["p10"], group_data[m_routing]["p90"], color=color_r, alpha=0.1)
                            else:
                                err_key = f"{stat_metric}_{self.error_bars}"
                                if err_key in group_data[m_routing]:
                                    plt.fill_between(layer_arr, m_arr - group_data[m_routing][err_key], m_arr + group_data[m_routing][err_key], color=color_r, alpha=0.15)
                                    
                    # Mixing norm
                    m_mixing = "mixing_weight_norm"
                    if m_mixing in group_data:
                        layer_arr = group_data[m_mixing]["layers"]
                        color_m = "navy"
                        for stat_metric in stat_metrics:
                            m_arr = group_data[m_mixing][stat_metric]
                            lbl = r"Mixing $\|W_V W_O\|_2$" if len(stat_metrics) == 1 else f"Mixing ({stat_metric.capitalize()})"
                            plt.plot(layer_arr, m_arr, marker='^', color=color_m, label=lbl, linewidth=2, markersize=4)
                            
                            # Error
                            if self.error_bars == "fan" or self.error_bars == "percentiles":
                                if "p10" in group_data[m_mixing] and "p90" in group_data[m_mixing]:
                                    plt.fill_between(layer_arr, group_data[m_mixing]["p10"], group_data[m_mixing]["p90"], color=color_m, alpha=0.1)
                            else:
                                err_key = f"{stat_metric}_{self.error_bars}"
                                if err_key in group_data[m_mixing]:
                                    plt.fill_between(layer_arr, m_arr - group_data[m_mixing][err_key], m_arr + group_data[m_mixing][err_key], color=color_m, alpha=0.15)
                                    
                    # Critical Baseline y = 1.0
                    plt.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, label='Neutral Boundary (y=1.0)')
                    
                    plt.title(f"Static Weight Amplifiers\n{self.group_titles[group_key]}", fontsize=12)
                    plt.xlabel("Layer Index", fontsize=11)
                    plt.ylabel("Spectral Norm (mean across heads)", fontsize=11)
                    plt.xscale(x_scale)
                    plt.yscale(y_scale)
                    plt.grid(True, alpha=0.3)
                    plt.legend(loc='best')
                    plt.tight_layout()
                    
                    safe_key = group_key.replace(" ", "_").replace("/", "-")
                    metric_str = "-".join(stat_metrics)
                    filename = f"attn_static_weights_{safe_key}_{metric_str}_xscale-{x_scale}_yscale-{y_scale}.png"
                    plt.savefig(os.path.join(self.plots_dir, filename), dpi=300)
                    plt.close()

    def plot_spectral_gaps(self, group_key):
        """Plot 4: Spectral Gap (The Contractive Force)"""
        group_data = self.data[group_key]
        x_scales = self.plotting_cfg.get("x_scales", ["linear"])
        y_scales = self.plotting_cfg.get("y_scales", ["linear"])
        
        metrics_to_process = [[m] for m in self.metric_list] if self.separate_figure_metrics else [self.metric_list]
        
        for x_scale in x_scales:
            for y_scale in y_scales:
                for stat_metrics in metrics_to_process:
                    plt.figure(figsize=(10, 6))
                    
                    for n in self.found_N_list:
                        m_name = f"mean_spectral_gap_N-{n}"
                        if m_name not in group_data:
                            continue
                            
                        layer_arr = group_data[m_name]["layers"]
                        color = self._get_color_for_N(n)
                        
                        for stat_metric in stat_metrics:
                            m_arr = group_data[m_name][stat_metric]
                            line_label = f"N = {n}" if len(stat_metrics) == 1 else f"N = {n} ({stat_metric.capitalize()})"
                            plt.plot(layer_arr, m_arr, marker='o', color=color, label=line_label, linewidth=2, markersize=4)
                            
                            # Error Shading
                            if self.error_bars == "fan" or self.error_bars == "percentiles":
                                if "p10" in group_data[m_name] and "p90" in group_data[m_name]:
                                    plt.fill_between(layer_arr, group_data[m_name]["p10"], group_data[m_name]["p90"], color=color, alpha=0.1)
                                if "p25" in group_data[m_name] and "p75" in group_data[m_name]:
                                    plt.fill_between(layer_arr, group_data[m_name]["p25"], group_data[m_name]["p75"], color=color, alpha=0.2)
                            else:
                                err_key = f"{stat_metric}_{self.error_bars}"
                                if err_key in group_data[m_name]:
                                    e_arr = group_data[m_name][err_key]
                                elif self.error_bars in group_data[m_name]:
                                    e_arr = group_data[m_name][self.error_bars]
                                else:
                                    e_arr = None
                                    
                                if e_arr is not None:
                                    lower = np.maximum(0, m_arr - e_arr)
                                    plt.fill_between(layer_arr, lower, m_arr + e_arr, color=color, alpha=0.2)
                                    
                    plt.title(f"Spectral Gap of Attention Matrix ($1 - \\sigma_2$)\n{self.group_titles[group_key]}", fontsize=12)
                    plt.xlabel("Layer Index", fontsize=11)
                    plt.ylabel("Mean Spectral Gap", fontsize=11)
                    plt.xscale(x_scale)
                    plt.yscale(y_scale)
                    plt.grid(True, alpha=0.3)
                    plt.legend(loc='best')
                    plt.tight_layout()
                    
                    safe_key = group_key.replace(" ", "_").replace("/", "-")
                    metric_str = "-".join(stat_metrics)
                    filename = f"attn_spectral_gaps_{safe_key}_{metric_str}_xscale-{x_scale}_yscale-{y_scale}.png"
                    plt.savefig(os.path.join(self.plots_dir, filename), dpi=300)
                    plt.close()

    def plot_token_sensitivity(self, group_key):
        """Plot 5: Spatial Sensitivity Profile (Token-Wise) at Peak Instability Layer"""
        group_data = self.data[group_key]
        
        # 1. Dynamically find the layer where spectral norm for max N is maximized
        max_n = max(self.found_N_list)
        norm_metric = f"attn_spectral_norm_N-{max_n}"
        
        if norm_metric not in group_data:
            print(f"Warning: Could not find norm metric {norm_metric} to identify peak layer.")
            return
            
        layers = group_data[norm_metric]["layers"]
        # Use primary stat metric to find peak (e.g. mean)
        stat_metric = self.metric_list[0]
        norm_vals = group_data[norm_metric][stat_metric]
        peak_idx = int(np.argmax(norm_vals))
        peak_layer = int(layers[peak_idx])
        
        print(f"Dynamic peak layer for sensitivity profile: Layer {peak_layer} (index {peak_idx})")
        
        # 2. Setup a figure with vertically stacked subplots for each N
        fig, axes = plt.subplots(len(self.found_N_list), 1, figsize=(10, 3 * len(self.found_N_list)), sharex=False)
        if len(self.found_N_list) == 1:
            axes = [axes]
            
        for idx, n in enumerate(self.found_N_list):
            ax = axes[idx]
            prof_metric = f"token_sensitivity_profile_N-{n}"
            if prof_metric not in group_data:
                ax.text(0.5, 0.5, f"Metric {prof_metric} not found", ha='center', va='center')
                continue
                
            # Extract profile for the peak layer
            # group_data[prof_metric][stat_metric] has shape [num_layers, n]
            profile_mean = group_data[prof_metric][stat_metric][peak_idx]
            color = self._get_color_for_N(n)
            
            x_vals = np.arange(n)
            ax.plot(x_vals, profile_mean, color=color, linewidth=1.5, label=f"N = {n}")
            
            # Error Shading
            if self.error_bars == "fan" or self.error_bars == "percentiles":
                if "p10" in group_data[prof_metric] and "p90" in group_data[prof_metric]:
                    p10_prof = group_data[prof_metric]["p10"][peak_idx]
                    p90_prof = group_data[prof_metric]["p90"][peak_idx]
                    ax.fill_between(x_vals, p10_prof, p90_prof, color=color, alpha=0.1)
                if "p25" in group_data[prof_metric] and "p75" in group_data[prof_metric]:
                    p25_prof = group_data[prof_metric]["p25"][peak_idx]
                    p75_prof = group_data[prof_metric]["p75"][peak_idx]
                    ax.fill_between(x_vals, p25_prof, p75_prof, color=color, alpha=0.2)
            else:
                err_key = f"{stat_metric}_{self.error_bars}"
                if err_key in group_data[prof_metric]:
                    err_prof = group_data[prof_metric][err_key][peak_idx]
                    ax.fill_between(x_vals, np.maximum(0, profile_mean - err_prof), profile_mean + err_prof, color=color, alpha=0.2)
                    
            ax.set_title(f"Sensitivity Profile for N = {n}", fontsize=10)
            ax.set_ylabel("Relative Sensitivity", fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right')
            
        axes[-1].set_xlabel("Token Sequence Index", fontsize=11)
        fig.suptitle(f"Token-Wise Spatial Sensitivity Profile at Peak Layer {peak_layer}\n{self.group_titles[group_key]}", fontsize=12)
        plt.tight_layout()
        
        safe_key = group_key.replace(" ", "_").replace("/", "-")
        filename = f"attn_token_sensitivity_profile_layer-{peak_layer}_{safe_key}.png"
        plt.savefig(os.path.join(self.plots_dir, filename), dpi=300)
        plt.close()

    def plot_all(self):
        print("--- Generating Attention Jacobian Plots ---")
        for group_key in self.data.keys():
            if not self._should_plot(group_key): 
                continue
            print(f"Plotting for group: {group_key}")
            self.plot_spectral_norms(group_key)
            self.plot_attention_entropy(group_key)
            self.plot_static_weights(group_key)
            self.plot_spectral_gaps(group_key)
            self.plot_token_sensitivity(group_key)
        print("-" * 50)

def main():
    config_path = "jacobian_config.yaml"
    if not os.path.exists(config_path):
        print(f"Error: {config_path} not found.")
        return
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    results_dir = config.get("experiment", {}).get("results_dir", "./results_jacobians_microsoft")
    plotting_cfg = config.get("plotting", {})
    
    data_path = os.path.join(results_dir, "analyzed_jacobians_attn.pkl")
    if not os.path.exists(data_path):
        print(f"Error: Analyzed data {data_path} not found.")
        return
        
    plotter = AttentionJacobianPlotter(data_path, plotting_cfg)
    plotter.plot_all()

if __name__ == "__main__":
    main()
