import os
import yaml
import pickle
import numpy as np
import matplotlib.pyplot as plt

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

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
    
    for metric_name in metric_list:
        print(f"Generating plots for aggregation metric: {metric_name} (error bars: {error_bars})...")
        
        # 1. Figure 1: Linearity Sweep (Log-Log Plot)
        for norm_name in ["rms_actual", "rms_pure", "layernorm_pure"]:
            # Find active combinations for this norm name
            has_orth = any(k[0] == norm_name and k[1] == "orthogonal" for k in aggregated_sweep.keys())
            if not has_orth:
                continue
                
            plt.figure(figsize=(10, 8))
            sweep_radii = sorted(radii)
            
            for pert_type, color, marker, label_name in [
                ("orthogonal", "navy", "o", "Orthogonal"), 
                ("radial", "crimson", "s", "Radial")
            ]:
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
                
                # Plot empirical finite difference
                plt.plot(sweep_radii, emp_vals, marker=marker, color=color, linewidth=2.5, 
                         label=f"Empirical {label_name}{exp_label}")
                
                # Proper error propagation shading
                valid_mask = ~np.isnan(emp_vals)
                if error_bars in ["fan", "percentiles"]:
                    valid_fan = ~np.isnan(p10_vals) & ~np.isnan(p90_vals)
                    plt.fill_between(np.array(sweep_radii)[valid_fan], p10_vals[valid_fan], p90_vals[valid_fan], 
                                     color=color, alpha=0.1)
                    plt.fill_between(np.array(sweep_radii)[valid_fan], p25_vals[valid_fan], p75_vals[valid_fan], 
                                     color=color, alpha=0.2)
                elif error_bars == "std":
                    valid_err = valid_mask & ~np.isnan(err_vals)
                    lower = np.maximum(1e-15, emp_vals - err_vals)
                    plt.fill_between(np.array(sweep_radii)[valid_err], lower[valid_err], (emp_vals + err_vals)[valid_err], 
                                     color=color, alpha=0.15)
                                 
                # Plot JVP (Option 2 - Linear baseline)
                plt.plot(sweep_radii, jvp_vals, linestyle="--", color=color, alpha=0.6, 
                         label=f"JVP {label_name}")
                         
            # Guide line representing slope of 1.0 (perfect linear scaling)
            guide_x = np.array([1e-8, 1.0])
            ref_key = (norm_name, "orthogonal", target_radius)
            ref_y = aggregated_sweep[ref_key]["emp_norm"][metric_name]
            guide_y = guide_x * (ref_y / target_radius)
            plt.plot(guide_x, guide_y, color="gray", linestyle=":", label="Slope = 1.0")
            
            plt.xscale("log")
            plt.yscale("log")
            plt.xlabel("Perturbation Magnitude $\epsilon = \|\delta x\|_2$")
            plt.ylabel(f"Output Perturbation Norm $\|\delta y\|_2$ ({metric_name.capitalize()})")
            plt.grid(True, which="both", alpha=0.3)
            plt.legend(loc="lower right")
            plt.tight_layout()
            
            plot_name = f"normalization_sweep_{norm_name}_{metric_name}.png"
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
                     label=f"JVP Scaling Ratio ({metric_name.capitalize()})")
            
            # Error propagation shading
            if error_bars in ["fan", "percentiles"]:
                plt.fill_between(layers_arr, jvp_p10, jvp_p90, color="navy", alpha=0.1)
                plt.fill_between(layers_arr, jvp_p25, jvp_p75, color="navy", alpha=0.2)
            elif error_bars == "std":
                lower = np.maximum(0.0, jvp_vals - jvp_errs)
                plt.fill_between(layers_arr, lower, jvp_vals + jvp_errs, color="navy", alpha=0.15)
            
            # Plot Empirical ratio
            plt.plot(layers_arr, emp_vals, marker="s", color="darkorange", linewidth=1.5, linestyle=":", 
                     label=f"Empirical Scaling Ratio ($\epsilon$={target_radius:.1e})")
                     
            plt.xlabel("Layer Index")
            plt.ylabel(f"Scaling Gain $\|\delta y\|_2 / \|\delta x_{\perp}\|_2$ ({metric_name.capitalize()})")
            plt.grid(True, alpha=0.3)
            plt.legend(loc="best")
            plt.tight_layout()
            
            plot_name = f"normalization_scaling_verification_{norm_name}_{metric_name}.png"
            plt.savefig(os.path.join(plots_dir, plot_name), dpi=dpi)
            plt.close()
            print(f"Saved radial scaling verification to {os.path.join(plots_dir, plot_name)}")
            
        # 3. Figure 3: Orthogonal Annihilation Verification (Cosine Similarity vs Layer)
        for norm_name in ["rms_actual", "rms_pure", "layernorm_pure"]:
            plt.figure(figsize=(10, 6))
            
            plotted = False
            for pert_type, color, marker, label_name in [
                ("orthogonal", "navy", "o", "Orthogonal Perturbation"), 
                ("radial", "crimson", "s", "Radial Perturbation")
            ]:
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
                         label=f"{label_name} ({metric_name.capitalize()})")
                
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
            
            plt.xlabel("Layer Index")
            plt.ylabel(r"Alignment $\cos(\theta) = \frac{x^T \delta y}{\|x\|_2 \|\delta y\|_2}$")
            plt.ylim(-0.2, 0.2)
            plt.grid(True, alpha=0.3)
            plt.legend(loc="best")
            plt.tight_layout()
            
            plot_name = f"normalization_orthogonal_annihilation_{norm_name}_{metric_name}.png"
            plt.savefig(os.path.join(plots_dir, plot_name), dpi=dpi)
            plt.close()
            print(f"Saved orthogonal annihilation verification to {os.path.join(plots_dir, plot_name)}")

if __name__ == "__main__":
    main()
