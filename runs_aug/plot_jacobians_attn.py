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
        
        # Expose and apply global styling configuration
        self.dpi = self.plotting_cfg.get("dpi", 300)
        font_size = self.plotting_cfg.get("font_size", 14)
        label_size = self.plotting_cfg.get("label_size", 14)
        tick_size = self.plotting_cfg.get("tick_size", 12)
        legend_size = self.plotting_cfg.get("legend_size", 11)
        
        plt.rcParams.update({
            "figure.dpi": self.dpi,
            "savefig.dpi": self.dpi,
            "font.size": font_size,
            "axes.labelsize": label_size,
            "xtick.labelsize": tick_size,
            "ytick.labelsize": tick_size,
            "legend.fontsize": legend_size,
            "font.family": "DejaVu Serif",
        })
        
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
        if n == 20:
            return "#1f77b4" # Blue
        elif n == 100:
            return "#ff7f0e" # Orange
        elif n == 1000:
            return "#2ca02c" # Green
        else:
            return plt.cm.tab10(self.found_N_list.index(n) % 10)
            
    def plot_spectral_norms(self, group_key):
        """Plot 1: Attention Global Spectral Norm"""
        group_data = self.data[group_key]
        x_scales = self.plotting_cfg.get("x_scales", ["linear"])
        y_scales = self.plotting_cfg.get("y_scales", ["linear"])
        
        metrics_to_process = [[m] for m in self.metric_list] if self.separate_figure_metrics else [self.metric_list]
        info_key = get_safe_filename_info(group_key, self.group_titles)
        
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
                    
                    if self.plotting_cfg.get("show_title", False):
                        plt.title(r"Attention Jacobian Global Spectral Norm $\|J_{attn}\|_2$" + f"\n{self.group_titles[group_key]}")
                    plt.xlabel("Layer")
                    plt.ylabel(r"$\|J_{attn}\|_2$")
                    plt.xscale(x_scale)
                    plt.yscale(y_scale)
                    plt.grid(True, alpha=0.3)
                    
                    # Suppress legend if a single curve is plotted
                    if len(self.found_N_list) * len(stat_metrics) > 1:
                        plt.legend(loc='best')
                    plt.tight_layout()
                    
                    metric_str = "-".join(stat_metrics)
                    filename = f"attn_spectral_norms_{info_key}_{metric_str}_xscale-{x_scale}_yscale-{y_scale}.png"
                    plt.savefig(os.path.join(self.plots_dir, filename), dpi=self.dpi)
                    plt.close()

    def plot_attention_entropy(self, group_key):
        """Plot 2: Dynamic Attention Entropy"""
        group_data = self.data[group_key]
        x_scales = self.plotting_cfg.get("x_scales", ["linear"])
        y_scales = self.plotting_cfg.get("y_scales", ["linear"])
        
        metrics_to_process = [[m] for m in self.metric_list] if self.separate_figure_metrics else [self.metric_list]
        info_key = get_safe_filename_info(group_key, self.group_titles)
        
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
                                    
                        # Draw exact theoretical max entropy line for causal attention
                        max_ent = np.mean(np.log2(np.arange(1, n + 1)))
                        plt.axhline(y=max_ent, color=color, linestyle='--', alpha=0.5, label=f"Max Causal H (N={n}): {max_ent:.2f}")
                        
                    if self.plotting_cfg.get("show_title", False):
                        plt.title(f"Dynamic Attention Entropy (Shannon)\n{self.group_titles[group_key]}")
                    plt.xlabel("Layer")
                    plt.ylabel("Entropy (bits)")
                    plt.xscale(x_scale)
                    plt.yscale(y_scale)
                    plt.grid(True, alpha=0.3)
                    
                    # Suppress legend if a single curve is plotted
                    if len(self.found_N_list) * len(stat_metrics) > 1:
                        plt.legend(loc='best')
                    plt.tight_layout()
                    
                    metric_str = "-".join(stat_metrics)
                    filename = f"attn_entropy_{info_key}_{metric_str}_xscale-{x_scale}_yscale-{y_scale}.png"
                    plt.savefig(os.path.join(self.plots_dir, filename), dpi=self.dpi)
                    plt.close()

    def plot_static_weights(self, group_key):
        """Plot 3: Static Weight Amplifiers"""
        group_data = self.data[group_key]
        x_scales = self.plotting_cfg.get("x_scales", ["linear"])
        y_scales = self.plotting_cfg.get("y_scales", ["linear"])
        
        metrics_to_process = [[m] for m in self.metric_list] if self.separate_figure_metrics else [self.metric_list]
        info_key = get_safe_filename_info(group_key, self.group_titles)
        
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
                    
                    if self.plotting_cfg.get("show_title", False):
                        plt.title(f"Static Weight Amplifiers\n{self.group_titles[group_key]}")
                    plt.xlabel("Layer")
                    plt.ylabel("Weight Spectral Norm")
                    plt.xscale(x_scale)
                    plt.yscale(y_scale)
                    plt.grid(True, alpha=0.3)
                    
                    # Suppress legend if a single curve is plotted
                    total_curves = (1 if m_routing in group_data else 0) + (1 if m_mixing in group_data else 0)
                    if total_curves * len(stat_metrics) > 1:
                        plt.legend(loc='best')
                    plt.tight_layout()
                    
                    metric_str = "-".join(stat_metrics)
                    filename = f"attn_static_weights_{info_key}_{metric_str}_xscale-{x_scale}_yscale-{y_scale}.png"
                    plt.savefig(os.path.join(self.plots_dir, filename), dpi=self.dpi)
                    plt.close()

    def plot_spectral_gaps(self, group_key):
        """Plot 4: Spectral Gap (The Contractive Force)"""
        group_data = self.data[group_key]
        x_scales = self.plotting_cfg.get("x_scales", ["linear"])
        y_scales = self.plotting_cfg.get("y_scales", ["linear"])
        
        metrics_to_process = [[m] for m in self.metric_list] if self.separate_figure_metrics else [self.metric_list]
        info_key = get_safe_filename_info(group_key, self.group_titles)
        
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
                                    
                    if self.plotting_cfg.get("show_title", False):
                        plt.title(f"Spectral Gap of Attention Matrix ($1 - \\sigma_2$)\n{self.group_titles[group_key]}")
                    plt.xlabel("Layer")
                    plt.ylabel("Spectral Gap")
                    plt.xscale(x_scale)
                    plt.yscale(y_scale)
                    plt.grid(True, alpha=0.3)
                    
                    # Suppress legend if a single curve is plotted
                    if len(self.found_N_list) * len(stat_metrics) > 1:
                        plt.legend(loc='best')
                    plt.tight_layout()
                    
                    metric_str = "-".join(stat_metrics)
                    filename = f"attn_spectral_gaps_{info_key}_{metric_str}_xscale-{x_scale}_yscale-{y_scale}.png"
                    plt.savefig(os.path.join(self.plots_dir, filename), dpi=self.dpi)
                    plt.close()

    def plot_token_sensitivity(self, group_key):
        """Plot 5: Spatial Sensitivity Profile (Token-Wise) at Peak Instability Layer"""
        group_data = self.data[group_key]
        
        # 1. Dynamically find the layer where spectral norm for the maximum completed N is maximized
        completed_Ns = [n for n in self.found_N_list if f"attn_spectral_norm_N-{n}" in group_data]
        if not completed_Ns:
            print(f"Warning: Could not find any completed norm metrics to identify peak layer.")
            return
            
        max_completed_n = max(completed_Ns)
        norm_metric = f"attn_spectral_norm_N-{max_completed_n}"
        
        layers = group_data[norm_metric]["layers"]
        stat_metric = self.metric_list[0]
        norm_vals = group_data[norm_metric][stat_metric]
        peak_idx = int(np.argmax(norm_vals))
        peak_layer = int(layers[peak_idx])
        
        print(f"Dynamic peak layer for sensitivity profile: Layer {peak_layer} (index {peak_idx})")
        
        # 2. Setup a figure with vertically stacked subplots for each completed N
        fig, axes = plt.subplots(len(completed_Ns), 1, figsize=(10, 3 * len(completed_Ns)), sharex=False)
        if len(completed_Ns) == 1:
            axes = [axes]
            
        for idx, n in enumerate(completed_Ns):
            ax = axes[idx]
            prof_metric = f"token_sensitivity_profile_N-{n}"
            if prof_metric not in group_data:
                ax.text(0.5, 0.5, f"Metric {prof_metric} not found", ha='center', va='center')
                continue
                
            prof_stat_metric = stat_metric if stat_metric in ["mean", "median"] else "mean"
            profile_mean = group_data[prof_metric][prof_stat_metric][peak_idx]
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
                err_key = f"{prof_stat_metric}_{self.error_bars}"
                if err_key in group_data[prof_metric]:
                    err_prof = group_data[prof_metric][err_key][peak_idx]
                    ax.fill_between(x_vals, np.maximum(0, profile_mean - err_prof), profile_mean + err_prof, color=color, alpha=0.2)
                    
            if self.plotting_cfg.get("show_title", False):
                ax.set_title(f"Sensitivity Profile for N = {n}")
            ax.set_ylabel("Sensitivity")
            ax.grid(True, alpha=0.3)
            
        axes[-1].set_xlabel("Token")
        if self.plotting_cfg.get("show_title", False):
            fig.suptitle(f"Token-Wise Spatial Sensitivity Profile at Peak Layer {peak_layer}\n{self.group_titles[group_key]}")
        plt.tight_layout()
        
        safe_key = get_safe_filename_info(group_key, self.group_titles)
        filename = f"attn_token_sensitivity_profile_layer-{peak_layer}_{safe_key}.png"
        plt.savefig(os.path.join(self.plots_dir, filename), dpi=self.dpi)
        plt.close()

    def plot_weight_alignment(self, group_key):
        """Plot 6: Singular Vector-Weight Alignment Index"""
        group_data = self.data[group_key]
        x_scales = self.plotting_cfg.get("x_scales", ["linear"])
        y_scales = ["linear"] # Correlation/Cosine similarity is bounded [0, 1]
        
        metrics_to_process = [[m] for m in self.metric_list] if self.separate_figure_metrics else [self.metric_list]
        info_key = get_safe_filename_info(group_key, self.group_titles)
        
        for x_scale in x_scales:
            for y_scale in y_scales:
                for stat_metrics in metrics_to_process:
                    plt.figure(figsize=(10, 6))
                    
                    for n in self.found_N_list:
                        m_name = f"weight_alignment_index_N-{n}"
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
                                    upper = np.minimum(1.0, m_arr + e_arr)
                                    plt.fill_between(layer_arr, lower, upper, color=color, alpha=0.2)
                                    
                    if self.plotting_cfg.get("show_title", False):
                        plt.title(f"Singular Vector-Weight Alignment ($\\alpha_\\ell$)\n{self.group_titles[group_key]}")
                    plt.xlabel("Layer")
                    plt.ylabel(r"Alignment Index ($\alpha_\ell$)")
                    plt.ylim(0, 1.05)
                    plt.xscale(x_scale)
                    plt.yscale(y_scale)
                    plt.grid(True, alpha=0.3)
                    
                    if len(self.found_N_list) * len(stat_metrics) > 1:
                        plt.legend(loc='best')
                    plt.tight_layout()
                    
                    metric_str = "-".join(stat_metrics)
                    filename = f"attn_weight_alignment_{info_key}_{metric_str}_xscale-{x_scale}_yscale-{y_scale}.png"
                    plt.savefig(os.path.join(self.plots_dir, filename), dpi=self.dpi)
                    plt.close()

    def plot_attention_entropy_ratio(self, group_key):
        """Plot 7: Dynamic Attention Entropy Ratio"""
        group_data = self.data[group_key]
        x_scales = self.plotting_cfg.get("x_scales", ["linear"])
        y_scales = ["linear"] # Ratio is [0, 1] bounded
        
        metrics_to_process = [[m] for m in self.metric_list] if self.separate_figure_metrics else [self.metric_list]
        info_key = get_safe_filename_info(group_key, self.group_titles)
        
        for x_scale in x_scales:
            for y_scale in y_scales:
                for stat_metrics in metrics_to_process:
                    plt.figure(figsize=(10, 6))
                    
                    for n in self.found_N_list:
                        m_name = f"entropy_ratio_N-{n}"
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
                                    upper = np.minimum(1.0, m_arr + e_arr)
                                    plt.fill_between(layer_arr, lower, upper, color=color, alpha=0.2)
                                    
                    if self.plotting_cfg.get("show_title", False):
                        plt.title(f"Dynamic Attention Entropy Ratio\n{self.group_titles[group_key]}")
                    plt.xlabel("Layer")
                    plt.ylabel(r"Entropy Ratio ($H / H_{max}$)")
                    plt.ylim(0, 1.05)
                    plt.xscale(x_scale)
                    plt.yscale(y_scale)
                    plt.grid(True, alpha=0.3)
                    
                    if len(self.found_N_list) * len(stat_metrics) > 1:
                        plt.legend(loc='best')
                    plt.tight_layout()
                    
                    metric_str = "-".join(stat_metrics)
                    filename = f"attn_entropy_ratio_{info_key}_{metric_str}_xscale-{x_scale}_yscale-{y_scale}.png"
                    plt.savefig(os.path.join(self.plots_dir, filename), dpi=self.dpi)
                    plt.close()

    def plot_x_norm_mean(self, group_key):
        """Plot 8: Hidden State Magnitude (x_norm_mean)"""
        group_data = self.data[group_key]
        x_scales = self.plotting_cfg.get("x_scales", ["linear"])
        y_scales = self.plotting_cfg.get("y_scales", ["linear"])
        
        metrics_to_process = [[m] for m in self.metric_list] if self.separate_figure_metrics else [self.metric_list]
        info_key = get_safe_filename_info(group_key, self.group_titles)
        
        for x_scale in x_scales:
            for y_scale in y_scales:
                for stat_metrics in metrics_to_process:
                    plt.figure(figsize=(10, 6))
                    
                    for n in self.found_N_list:
                        m_name = f"x_norm_mean_N-{n}"
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
                                    
                    if self.plotting_cfg.get("show_title", False):
                        plt.title(f"Hidden State Magnitude (Mean $L_2$ Norm)\n{self.group_titles[group_key]}")
                    plt.xlabel("Layer")
                    plt.ylabel(r"Mean $\|x_i\|_2$")
                    plt.xscale(x_scale)
                    plt.yscale(y_scale)
                    plt.grid(True, alpha=0.3)
                    
                    if len(self.found_N_list) * len(stat_metrics) > 1:
                        plt.legend(loc='best')
                    plt.tight_layout()
                    
                    metric_str = "-".join(stat_metrics)
                    filename = f"attn_x_norm_mean_{info_key}_{metric_str}_xscale-{x_scale}_yscale-{y_scale}.png"
                    plt.savefig(os.path.join(self.plots_dir, filename), dpi=self.dpi)
                    plt.close()

    def plot_token_sensitivity_heatmap(self, group_key):
        """Plot 9: Token Sensitivity Heatmap (Layer vs Token)"""
        group_data = self.data[group_key]
        info_key = get_safe_filename_info(group_key, self.group_titles)
        metrics_to_process = [[m] for m in self.metric_list] if self.separate_figure_metrics else [self.metric_list]
        
        for stat_metrics in metrics_to_process:
            stat_metric = stat_metrics[0] # Heatmap only shows 1 metric at a time
            # Fallback to mean because harmonic mode is not computed for vectors
            plot_metric = stat_metric if stat_metric in ["mean", "median"] else "mean"
            for n in self.found_N_list:
                m_name = f"token_sensitivity_profile_N-{n}"
                if m_name not in group_data:
                    continue
                    
                layers = group_data[m_name]["layers"]
                # Shape is (num_layers, n)
                profile_matrix = np.array(group_data[m_name][plot_metric])
                
                # Double-check that it successfully loaded a 2D matrix
                if profile_matrix.ndim != 2:
                    print(f"Warning: Expected 2D matrix for heatmap, got {profile_matrix.shape}. Skipping.")
                    continue
                
                plt.figure(figsize=(12, 8))
                extent = [-0.5, n - 0.5, layers[-1] + 0.5, layers[0] - 0.5]
                
                im = plt.imshow(profile_matrix, aspect='auto', cmap='magma', extent=extent, interpolation='nearest')
                plt.colorbar(im, label=f"Sensitivity ({plot_metric.capitalize()})")
                
                if self.plotting_cfg.get("show_title", False):
                    plt.title(f"Token Sensitivity Heatmap N={n}\n{self.group_titles[group_key]}")
                    
                plt.xlabel("Token Index")
                plt.ylabel("Layer")
                plt.gca().invert_yaxis()
                
                plt.tight_layout()
                metric_str = "-".join(stat_metrics)
                filename = f"attn_token_sensitivity_heatmap_{info_key}_N-{n}_{metric_str}.png"
                plt.savefig(os.path.join(self.plots_dir, filename), dpi=self.dpi)
                plt.close()

    def plot_token_sensitivity_swarm(self, group_key):
        """Plot 10: Swarm Plot of Token Sensitivities per Layer"""
        group_data = self.data[group_key]
        x_scales = self.plotting_cfg.get("x_scales", ["linear"])
        y_scales = self.plotting_cfg.get("y_scales", ["log"])
        
        metrics_to_process = [[m] for m in self.metric_list] if self.separate_figure_metrics else [self.metric_list]
        info_key = get_safe_filename_info(group_key, self.group_titles)
        
        for x_scale in x_scales:
            for y_scale in y_scales:
                for stat_metrics in metrics_to_process:
                    stat_metric = stat_metrics[0]
                    plot_metric = stat_metric if stat_metric in ["mean", "median"] else "mean"
                    for n in self.found_N_list:
                        m_name = f"token_sensitivity_profile_N-{n}"
                        if m_name not in group_data or "raw" not in group_data[m_name]:
                            continue
                            
                        layers = group_data[m_name]["layers"]
                        raw_data = group_data[m_name]["raw"] # List of shape (P, n) arrays
                        profile_matrix = np.array(group_data[m_name][plot_metric])
                        
                        if profile_matrix.ndim == 1 and profile_matrix.size > 0 and not isinstance(profile_matrix[0], (list, np.ndarray)):
                            # Safety check for invalid fallback
                            layer_means = np.zeros(len(layers))
                        else:
                            layer_means = profile_matrix.mean(axis=1) if profile_matrix.ndim == 2 else profile_matrix
                        
                        plt.figure(figsize=(12, 6))
                        
                        for i, layer in enumerate(layers):
                            layer_data = raw_data[i].flatten()
                            
                            # Subsample if too many points to avoid incredibly slow plots
                            if len(layer_data) > 5000:
                                layer_data = np.random.choice(layer_data, 5000, replace=False)
                                
                            layer_data = np.maximum(layer_data, 1e-8)
                            x_jitter = layer + np.random.uniform(-0.25, 0.25, size=len(layer_data))
                            
                            plt.scatter(x_jitter, layer_data, s=2.5, alpha=0.5, color='teal')
                        
                        plt.plot(layers, layer_means, color='red', marker='D', markersize=5, linestyle='-', linewidth=2, label=f"{plot_metric.capitalize()} Sensitivity", zorder=10)
                        
                        if self.plotting_cfg.get("show_title", False):
                            plt.title(f"Token Sensitivity Distribution N={n}\n{self.group_titles[group_key]}")
                        plt.xlabel("Layer")
                        plt.ylabel("Sensitivity")
                        plt.xscale(x_scale)
                        plt.yscale(y_scale)
                        plt.grid(True, alpha=0.3, axis='y')
                        plt.legend()
                        
                        plt.tight_layout()
                        metric_str = "-".join(stat_metrics)
                        filename = f"attn_token_sensitivity_swarm_{info_key}_N-{n}_{metric_str}_xscale-{x_scale}_yscale-{y_scale}.png"
                        plt.savefig(os.path.join(self.plots_dir, filename), dpi=self.dpi)
                        plt.close()

    def plot_all(self):
        print("--- Generating Attention Jacobian Plots ---")
        for group_key in self.data.keys():
            if not self._should_plot(group_key): 
                continue
            print(f"Plotting for group: {group_key}")
            if self.plotting_cfg.get("plot_spectral_norms", True):
                self.plot_spectral_norms(group_key)
            if self.plotting_cfg.get("plot_attention_entropy", True):
                self.plot_attention_entropy(group_key)
            if self.plotting_cfg.get("plot_attention_entropy_ratio", False):
                self.plot_attention_entropy_ratio(group_key)
            if self.plotting_cfg.get("plot_static_weights", True):
                self.plot_static_weights(group_key)
            if self.plotting_cfg.get("plot_spectral_gaps", True):
                self.plot_spectral_gaps(group_key)
            if self.plotting_cfg.get("plot_token_sensitivity", True):
                self.plot_token_sensitivity(group_key)
            if self.plotting_cfg.get("plot_weight_alignment", True):
                self.plot_weight_alignment(group_key)
            if self.plotting_cfg.get("plot_x_norm_mean", True):
                self.plot_x_norm_mean(group_key)
            if self.plotting_cfg.get("plot_token_sensitivity_heatmap", True):
                self.plot_token_sensitivity_heatmap(group_key)
            if self.plotting_cfg.get("plot_swarm", True):
                self.plot_token_sensitivity_swarm(group_key)
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
