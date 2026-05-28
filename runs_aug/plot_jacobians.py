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
        
    def _plot_metric_across_layers(self, group_key, metric_names, title_prefix, ylabel, filename_prefix, labels=None, colors=None, hlines=None):
        metrics_to_process = [[m] for m in self.metric_list] if self.separate_figure_metrics else [self.metric_list]
        group_data = self.data[group_key]
        
        x_scales = self.plotting_cfg.get("x_scales", self.plotting_cfg.get("x_scale", ["linear"]))
        y_scales = self.plotting_cfg.get("y_scales", self.plotting_cfg.get("y_scale", ["linear"]))
        if isinstance(x_scales, str):
            x_scales = [x_scales]
        if isinstance(y_scales, str):
            y_scales = [y_scales]
            
        info_key = get_safe_filename_info(group_key, self.group_titles)
        total_curves = len(metric_names)
            
        for x_scale in x_scales:
            for y_scale in y_scales:
                for current_metric_list in metrics_to_process:
                    plt.figure(figsize=(10, 6))
                    if colors is None:
                        cmap = plt.cm.tab10(np.linspace(0, 1, max(total_curves * len(current_metric_list), 10)))
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
                            else:
                                err_key = f"{stat_metric}_{self.error_bars}"
                                if err_key in group_data[m_name]:
                                    e_arr = group_data[m_name][err_key]
                                elif self.error_bars in group_data[m_name]:
                                    e_arr = group_data[m_name][self.error_bars]
                                else:
                                    e_arr = None
                                    
                                if e_arr is not None:
                                    lower = np.maximum(0, m_arr - e_arr) if self.error_bars in ["std", "var"] else m_arr - e_arr
                                    if y_scale == "log":
                                        lower = np.maximum(1e-12, lower)
                                    plt.fill_between(layer_arr, lower, m_arr + e_arr, color=color, alpha=0.2)
                    
                    if hlines:
                        for h in hlines:
                            plt.axhline(y=h['y'], color=h.get('color', 'black'), linestyle=h.get('linestyle', '--'), label=h.get('label', ''))
                            
                    if self.plotting_cfg.get("show_title", False):
                        title = f"{title_prefix} | {self.group_titles[group_key]}"
                        plt.title(title)
                    plt.xlabel("Layer")
                    plt.ylabel(ylabel)
                    plt.xscale(x_scale)
                    plt.yscale(y_scale)
                    plt.grid(True, alpha=0.3)
                    
                    # Suppress legend if a single curve is plotted
                    if total_curves * len(current_metric_list) > 1:
                        plt.legend(title="Metric", loc='best')
                    plt.tight_layout()
                    
                    metric_str = "-".join(current_metric_list)
                    plt.savefig(os.path.join(self.plots_dir, f"{filename_prefix}_{info_key}_{metric_str}_xscale-{x_scale}_yscale-{y_scale}.png"), dpi=self.dpi)
                    plt.close()

    def _plot_spectral_norm_lambda_true_together(self, group_key):
        metrics_to_process = [[m] for m in self.metric_list] if self.separate_figure_metrics else [self.metric_list]
        group_data = self.data[group_key]
        
        x_scales = self.plotting_cfg.get("x_scales", self.plotting_cfg.get("x_scale", ["linear"]))
        y_scales = self.plotting_cfg.get("y_scales", self.plotting_cfg.get("y_scale", ["linear"]))
        if isinstance(x_scales, str):
            x_scales = [x_scales]
        if isinstance(y_scales, str):
            y_scales = [y_scales]
            
        info_key = get_safe_filename_info(group_key, self.group_titles)
        
        metric_names = ["spectral_norms", "lambda_true"]
        pretty_labels = {
            "spectral_norms": r"$\| \mathbf{J} \|_2$",
            "lambda_true": r"$\bar{\lambda}_{true}$"
        }
        colors = {
            "spectral_norms": "darkred",
            "lambda_true": "navy"
        }
        
        for x_scale in x_scales:
            for y_scale in y_scales:
                for current_metric_list in metrics_to_process:
                    plt.figure(figsize=(10, 6))
                    
                    for m_name in metric_names:
                        if m_name not in group_data: continue
                        layer_arr = group_data[m_name]["layers"]
                        color = colors[m_name]
                        
                        for stat_metric in current_metric_list:
                            m_arr = group_data[m_name][stat_metric]
                            
                            base_label = pretty_labels[m_name]
                            line_label = f"{base_label} ({stat_metric.capitalize()})" if len(current_metric_list) > 1 else base_label
                            plt.plot(layer_arr, m_arr, marker='o', color=color, label=line_label, linewidth=2, markersize=4)
                            
                            # Fan Shading / Error Bars
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
                                    lower = np.maximum(0, m_arr - e_arr) if self.error_bars in ["std", "var"] else m_arr - e_arr
                                    if y_scale == "log":
                                        lower = np.maximum(1e-12, lower)
                                    plt.fill_between(layer_arr, lower, m_arr + e_arr, color=color, alpha=0.2)
                                    
                    # Critical Baseline y = 1.0
                    plt.axhline(y=1.0, color='black', linestyle='--', linewidth=1.5, label='Neutral Boundary')
                    
                    if self.plotting_cfg.get("show_title", False):
                        plt.title(f"Jacobian Spectral Norm & Lambda True | {self.group_titles[group_key]}")
                    plt.xlabel("Layer")
                    plt.ylabel("Value")
                    plt.xscale(x_scale)
                    plt.yscale(y_scale)
                    plt.grid(True, alpha=0.3)
                    
                    plt.legend(loc='best')
                    plt.tight_layout()
                    
                    metric_str = "-".join(current_metric_list)
                    filename = f"jacobian_together_{info_key}_{metric_str}_xscale-{x_scale}_yscale-{y_scale}.png"
                    plt.savefig(os.path.join(self.plots_dir, filename), dpi=self.dpi)
                    plt.close()

    def _plot_mft_comparison(self, group_key):
        group_data = self.data[group_key]
        
        required = ["lambda_true", "W_gate_scaled_F2", "W_up_scaled_F2", "W_down_scaled_F2", "S_x_sq_mean", "D_x_sq_mean"]
        for req in required:
            if req not in group_data:
                print(f"Skipping MFT comparison for {group_key} because {req} is missing.")
                return
                
        layers = group_data["lambda_true"]["layers"]
        
        y_true = group_data["lambda_true"]["mean"]
        w_gate = group_data["W_gate_scaled_F2"]["mean"]
        w_up = group_data["W_up_scaled_F2"]["mean"]
        w_down = group_data["W_down_scaled_F2"]["mean"]
        mean_S = group_data["S_x_sq_mean"]["mean"]
        mean_D = group_data["D_x_sq_mean"]["mean"]
        
        y_predicted = w_down * ((w_gate * mean_D) + (w_up * mean_S))
        
        # Propagate token-by-token error distribution
        y_pred_p10 = []
        y_pred_p25 = []
        y_pred_p75 = []
        y_pred_p90 = []
        
        has_raw = ("raw" in group_data["S_x_sq_mean"] and "raw" in group_data["D_x_sq_mean"])
        if has_raw:
            for idx in range(len(layers)):
                s_raw = group_data["S_x_sq_mean"]["raw"][idx]
                d_raw = group_data["D_x_sq_mean"]["raw"][idx]
                
                min_len = min(len(s_raw), len(d_raw))
                s_raw = s_raw[:min_len]
                d_raw = d_raw[:min_len]
                
                w_g = w_gate[idx]
                w_u = w_up[idx]
                w_d = w_down[idx]
                
                pred_raw = w_d * (w_g * d_raw + w_u * s_raw)
                p10, p25, p75, p90 = np.percentile(pred_raw, [10, 25, 75, 90])
                
                y_pred_p10.append(p10)
                y_pred_p25.append(p25)
                y_pred_p75.append(p75)
                y_pred_p90.append(p90)
                
            y_pred_p10 = np.array(y_pred_p10)
            y_pred_p25 = np.array(y_pred_p25)
            y_pred_p75 = np.array(y_pred_p75)
            y_pred_p90 = np.array(y_pred_p90)
            
        x_scales = self.plotting_cfg.get("x_scales", self.plotting_cfg.get("x_scale", ["linear"]))
        y_scales = self.plotting_cfg.get("y_scales", self.plotting_cfg.get("y_scale", ["linear"]))
        if isinstance(x_scales, str):
            x_scales = [x_scales]
        if isinstance(y_scales, str):
            y_scales = [y_scales]
            
        info_key = get_safe_filename_info(group_key, self.group_titles)
        
        for x_scale in x_scales:
            for y_scale in y_scales:
                plt.figure(figsize=(10, 6))
                
                # Empirical curve
                plt.plot(layers, y_true, marker='o', linestyle='-', color='b', label='Empirical', linewidth=2, markersize=5)
                
                # MFT curve
                plt.plot(layers, y_predicted, marker='s', linestyle='--', color='r', label='Mean Field', linewidth=2, markersize=5)
                
                # Fan shading
                if self.error_bars == "fan" or self.error_bars == "percentiles":
                    p10_emp = group_data["lambda_true"]["p10"]
                    p25_emp = group_data["lambda_true"]["p25"]
                    p75_emp = group_data["lambda_true"]["p75"]
                    p90_emp = group_data["lambda_true"]["p90"]
                    
                    p10_mft = y_pred_p10
                    p25_mft = y_pred_p25
                    p75_mft = y_pred_p75
                    p90_mft = y_pred_p90
                    
                    if y_scale == "log":
                        p10_emp = np.maximum(1e-12, p10_emp)
                        p25_emp = np.maximum(1e-12, p25_emp)
                        if has_raw:
                            p10_mft = np.maximum(1e-12, p10_mft)
                            p25_mft = np.maximum(1e-12, p25_mft)
                            
                    # Empirical Shading
                    plt.fill_between(layers, p10_emp, p90_emp, color='b', alpha=0.08)
                    plt.fill_between(layers, p25_emp, p75_emp, color='b', alpha=0.15)
                    
                    # MFT Shading
                    if has_raw:
                        plt.fill_between(layers, p10_mft, p90_mft, color='r', alpha=0.08)
                        plt.fill_between(layers, p25_mft, p75_mft, color='r', alpha=0.15)
                
                plt.axhline(y=1.0, color='black', linestyle=':', linewidth=1.5, label='Neutral Boundary')
                
                if self.plotting_cfg.get("show_title", False):
                    plt.title(f"MFT Stretching Factor Comparison | {self.group_titles[group_key]}")
                plt.xlabel("Layer")
                plt.ylabel(r"Mean Squared Singular Value ($\bar{\lambda}$)")
                
                plt.xscale(x_scale)
                plt.yscale(y_scale)
                plt.grid(True, which="both", alpha=0.3)
                
                plt.legend(loc='lower left')
                plt.tight_layout()
                
                filename = f"mft_comparison_{info_key}_xscale-{x_scale}_yscale-{y_scale}.png"
                plt.savefig(os.path.join(self.plots_dir, filename), dpi=self.dpi)
                plt.close()

    def _save_weight_averages(self, group_key):
        group_data = self.data[group_key]
        metrics = {
            "W_gate_max_SVD": "W_gate_SVD",
            "W_up_max_SVD": "W_up_SVD",
            "W_down_max_SVD": "W_down_SVD",
            "W_gate_scaled_F2": "W_gate_Frobenius",
            "W_up_scaled_F2": "W_up_Frobenius",
            "W_down_scaled_F2": "W_down_Frobenius"
        }
        
        results = {}
        for metric_name, pretty_name in metrics.items():
            if metric_name in group_data:
                arr = group_data[metric_name]["mean"]
                results[pretty_name] = float(np.mean(arr))
                
        if not results:
            return
            
        info_key = get_safe_filename_info(group_key, self.group_titles)
        out_csv = os.path.join(self.plots_dir, f"weight_averages_{info_key}.csv")
        
        with open(out_csv, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Matrix", "SVD", "Frobenius"])
            writer.writerow(["W_gate", results.get("W_gate_SVD"), results.get("W_gate_Frobenius")])
            writer.writerow(["W_up", results.get("W_up_SVD"), results.get("W_up_Frobenius")])
            writer.writerow(["W_down", results.get("W_down_SVD"), results.get("W_down_Frobenius")])
            
        print(f"Saved weight averages CSV to {out_csv}")

    def plot_distribution_heatmaps(self):
        print("Generating Jacobian Distribution Heatmaps...")
        # Only for token metrics (which have distributions)
        token_metrics = ["spectral_norms", "lambda_true"]
        for group_key in self.data.keys():
            if not self._should_plot(group_key): continue
            group_data = self.data[group_key]
            info_key = get_safe_filename_info(group_key, self.group_titles)
            
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
                
                if self.plotting_cfg.get("show_title", False):
                    plt.title(f"Jacobian Distribution: {m_name} | {self.group_titles[group_key]}")
                plt.xlabel("Layer")
                
                # Clean up heatmap y axis label
                if m_name == "spectral_norms":
                    simple_ylabel = r"$\| \mathbf{J} \|_2$"
                elif m_name == "lambda_true":
                    simple_ylabel = r"$\bar{\lambda}_{true}$"
                else:
                    simple_ylabel = m_name.replace("_", " ").title()
                plt.ylabel(simple_ylabel)
                
                plt.legend(); plt.tight_layout()
                
                plt.savefig(os.path.join(self.plots_dir, f"heatmap_{m_name}_{info_key}.png"), dpi=self.dpi)
                plt.close()

    def plot_swarm_plots(self):
        print("Generating Jacobian Swarm/Trajectory Plots...")
        setups = {}
        for group_key in self.data.keys():
            if group_key.endswith("_aggregated"):
                setup_name = group_key.replace("_aggregated", "")
                setups[setup_name] = []
                
        for group_key in self.data.keys():
            if not group_key.endswith("_aggregated"):
                for setup_name in setups.keys():
                    if group_key.startswith(setup_name):
                        setups[setup_name].append(group_key)
                        
        from matplotlib.lines import Line2D
        
        for setup_name, prompt_keys in setups.items():
            if not prompt_keys: continue
            
            for m_name in ["spectral_norms", "lambda_true", "S_x_sq_mean", "D_x_sq_mean"]:
                x_scales = self.plotting_cfg.get("x_scales", ["linear"])
                y_scales = self.plotting_cfg.get("y_scales", ["linear"])
                
                for x_scale in x_scales:
                    for y_scale in y_scales:
                        plt.figure(figsize=(10, 6))
                        
                        colors = plt.cm.tab20(np.linspace(0, 1, len(prompt_keys)))
                        
                        for p_idx, p_key in enumerate(prompt_keys):
                            p_data = self.data[p_key]
                            if m_name not in p_data or "raw" not in p_data[m_name]: continue
                            
                            layers = p_data[m_name]["layers"]
                            raw_data = p_data[m_name]["raw"]
                            medians = p_data[m_name]["median"]
                            
                            color = colors[p_idx % len(colors)]
                            
                            # Jittered scatter plot per layer
                            all_x = []
                            all_y = []
                            for l_idx, layer_val in enumerate(layers):
                                tokens_val = raw_data[l_idx]
                                jitter = np.random.uniform(-0.15, 0.15, size=len(tokens_val))
                                all_x.extend(layer_val + jitter)
                                all_y.extend(tokens_val)
                                
                            plt.scatter(all_x, all_y, color=color, s=8, alpha=0.3, edgecolors='none')
                            plt.plot(layers, medians, color=color, linewidth=1.5, alpha=0.8)
                            
                        # Add Neutral Boundary baseline at y = 1.0 for spectral_norms and lambda_true
                        if m_name in ["spectral_norms", "lambda_true"]:
                            plt.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5)

                        if self.plotting_cfg.get("show_title", False):
                            plt.title(f"Jacobian {m_name} Layer-wise Swarm | Setup: {setup_name}")
                        plt.xlabel("Layer")
                        
                        if m_name == "spectral_norms":
                            simple_ylabel = r"$\| \mathbf{J} \|_2$"
                        elif m_name == "lambda_true":
                            simple_ylabel = r"$\bar{\lambda}_{true}$"
                        elif m_name == "S_x_sq_mean":
                            simple_ylabel = r"$S(x)^2$"
                        elif m_name == "D_x_sq_mean":
                            simple_ylabel = r"$D(x)^2$"
                        else:
                            simple_ylabel = m_name.replace("_", " ").title()
                        plt.ylabel(simple_ylabel)
                        
                        plt.xscale(x_scale)
                        plt.yscale(y_scale)
                        plt.grid(True, which="both", alpha=0.3)
                        
                        legend_elements = [
                            Line2D([0], [0], marker='o', color='gray', linestyle='none', markersize=5, label='Token Values (individual)'),
                            Line2D([0], [0], color='gray', linewidth=1.5, label='Prompt Medians')
                        ]
                        if m_name in ["spectral_norms", "lambda_true"]:
                            legend_elements.append(Line2D([0], [0], color='red', linestyle='--', linewidth=1.5, label='Neutral Boundary'))
                        
                        plt.legend(handles=legend_elements, loc='upper left')
                        plt.tight_layout()
                        
                        filename = f"swarm_{m_name}_{setup_name}_xscale-{x_scale}_yscale-{y_scale}.png"
                        plt.savefig(os.path.join(self.plots_dir, filename), dpi=self.dpi)
                        plt.close()

    def plot_all(self):
        print("--- Generating Jacobian Plots ---")
        for group_key in self.data.keys():
            if not self._should_plot(group_key): continue
            
            # Save weight averages CSV directly inside the plots directory
            self._save_weight_averages(group_key)
            
            # Individual plots
            if self.plotting_cfg.get("plot_spectral_norms", True):
                self._plot_metric_across_layers(group_key, ["spectral_norms"], r"Jacobian Spectral Norm $\| \mathbf{J}_{MLP} \|_2$", r"$\| \mathbf{J} \|_2$", "spectral_norms", labels=["Spectral Norm"], colors=['darkred'], hlines=[{'y': 1.0, 'label': 'Neutral Boundary'}])
            if self.plotting_cfg.get("plot_lambda_true", True):
                self._plot_metric_across_layers(group_key, ["lambda_true"], r"Mean Squared Singular Value $\bar{\lambda}_{true}$", r"$\bar{\lambda}_{true}$", "lambda_true", labels=[r"$\bar{\lambda}_{true}$"], colors=['navy'], hlines=[{'y': 1.0, 'label': 'Neutral Boundary'}])
            
            # Joint plot
            if self.plotting_cfg.get("spectral_norm_lambda_true_together", False):
                self._plot_spectral_norm_lambda_true_together(group_key)
                
            # MFT Comparison Plot
            if self.plotting_cfg.get("plot_mft_comparison", False):
                self._plot_mft_comparison(group_key)
                
            if self.plotting_cfg.get("plot_weight_svds", True):
                self._plot_metric_across_layers(group_key, ["W_gate_max_SVD", "W_up_max_SVD", "W_down_max_SVD"], "Weight Matrix Spectral Norms", "Weight Spectral Norm", "weight_svds", labels=[r'$W_{gate}$', r'$W_{up}$', r'$W_{down}$'], hlines=[{'y': 1.0, 'label': 'Neutral Boundary'}])
            if self.plotting_cfg.get("plot_scaled_frobenius", True):
                self._plot_metric_across_layers(group_key, ["W_gate_scaled_F2", "W_up_scaled_F2", "W_down_scaled_F2"], "Scaled Frobenius Norms", "Scaled Frobenius Norm", "scaled_frobenius", labels=[r'$W_{gate}$', r'$W_{up}$', r'$W_{down}$'], hlines=[{'y': 1.0, 'label': 'Neutral Boundary'}])
            if self.plotting_cfg.get("plot_activation_densities", True):
                self._plot_metric_across_layers(group_key, ["S_x_sq_mean", "D_x_sq_mean"], "Activation Densities", "Activation Density", "activation_densities", labels=[r'$S(x)^2$', r'$D(x)^2$'], colors=['teal', 'orange'])
        
        if self.plotting_cfg.get("plot_heatmap", True):
            self.plot_distribution_heatmaps()
        if self.plotting_cfg.get("plot_swarm", False):
            self.plot_swarm_plots()
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
