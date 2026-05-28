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
    linearity_boundaries = analysis_results["linearity_boundaries"]
    layer_metrics = analysis_results["layer_metrics"]
    
    plots_dir = os.path.join(results_dir, "aggregated_plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Configure global plotting parameters
    dpi = plotting_cfg.get("dpi", 300)
    font_size = plotting_cfg.get("font_size", 20)
    label_size = plotting_cfg.get("label_size", 20)
    tick_size = plotting_cfg.get("tick_size", 16)
    legend_size = plotting_cfg.get("legend_size", 16)
    
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
    
    unique_norm_names = sorted(list(set(k.split("_")[0] + "_" + k.split("_")[1] for k in layer_metrics.keys())))
    
    # 1. Figure 1: Linearity Sweep (Log-Log Plot)
    # We plot this for each norm name
    for norm_name in ["rms_actual", "rms_pure", "layernorm_pure"]:
        # Find active combinations for this norm name
        has_orth = f"{norm_name}_orthogonal_1e-08" in aggregated_sweep or f"{norm_name}_orthogonal_1e-8" in aggregated_sweep
        if not has_orth:
            # Try to match the key
            matched = False
            for k in aggregated_sweep.keys():
                if k.startswith(norm_name):
                    matched = True
                    break
            if not matched:
                continue
                
        plt.figure(figsize=(10, 8))
        
        # Radii and output norms
        sweep_radii = sorted(radii)
        
        for pert_type, color, marker, label_name in [
            ("orthogonal", "navy", "o", "Orthogonal"), 
            ("radial", "crimson", "s", "Radial")
        ]:
            emp_medians = []
            jvp_medians = []
            theory_medians = []
            
            p10_emp = []
            p90_emp = []
            
            for r in sweep_radii:
                key = f"{norm_name}_{pert_type}_{r}"
                if key not in aggregated_sweep:
                    # check with float formatting
                    key = f"{norm_name}_{pert_type}_{float(r)}"
                if key in aggregated_sweep:
                    item = aggregated_sweep[key]
                    emp_medians.append(item["emp_norm"]["median"])
                    jvp_medians.append(item["jvp_norm"]["median"])
                    theory_medians.append(item["theory_norm"]["median"])
                    p10_emp.append(item["emp_norm"]["p10"])
                    p90_emp.append(item["emp_norm"]["p90"])
                else:
                    emp_medians.append(np.nan)
                    jvp_medians.append(np.nan)
                    theory_medians.append(np.nan)
                    p10_emp.append(np.nan)
                    p90_emp.append(np.nan)
                    
            emp_medians = np.array(emp_medians)
            jvp_medians = np.array(jvp_medians)
            theory_medians = np.array(theory_medians)
            p10_emp = np.array(p10_emp)
            p90_emp = np.array(p90_emp)
            
            combo_key = f"{norm_name}_{pert_type}"
            exponent = power_laws.get(combo_key, {}).get("exponent", None)
            exp_label = f" (k={exponent:.3f})" if exponent is not None else ""
            
            # Plot empirical finite difference
            plt.plot(sweep_radii, emp_medians, marker=marker, color=color, linewidth=2.5, 
                     label=f"Empirical {label_name}{exp_label}")
            # Shading for token spread
            valid_mask = ~np.isnan(p10_emp) & ~np.isnan(p90_emp)
            plt.fill_between(np.array(sweep_radii)[valid_mask], np.array(p10_emp)[valid_mask], np.array(p90_emp)[valid_mask], 
                             color=color, alpha=0.1)
                             
            # Plot JVP (Option 2 - Linear baseline)
            # JVP should be a perfect straight line on log-log
            plt.plot(sweep_radii, jvp_medians, linestyle="--", color=color, alpha=0.6, 
                     label=f"JVP {label_name}")
                     
        # Guide line representing slope of 1.0 (perfect linear scaling)
        guide_x = np.array([1e-8, 1.0])
        # scale guide line to match the orthogonal response at 1e-4
        ref_idx = sweep_radii.index(target_radius)
        ref_y = aggregated_sweep[f"{norm_name}_orthogonal_{target_radius}"]["emp_norm"]["median"]
        guide_y = guide_x * (ref_y / target_radius)
        plt.plot(guide_x, guide_y, color="gray", linestyle=":", label="Slope = 1.0")
        
        # Mark the linearity boundary if it exists
        boundary_rad_orth = linearity_boundaries.get(f"{norm_name}_orthogonal")
        if boundary_rad_orth:
            plt.axvline(boundary_rad_orth, color="black", linestyle="-.", alpha=0.5, 
                        label=f"Orth. Non-Linearity Limit ({boundary_rad_orth:.1e})")
                        
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Perturbation Magnitude $\epsilon = \|\delta x\|_2$")
        plt.ylabel("Output Perturbation Norm $\|\delta y\|_2$")
        plt.grid(True, which="both", alpha=0.3)
        plt.legend(loc="lower right")
        plt.tight_layout()
        
        plot_name = f"normalization_sweep_{norm_name}.png"
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
        
        theory_medians = []
        jvp_medians = []
        emp_medians = []
        
        jvp_p10 = []
        jvp_p90 = []
        
        for l in sorted_layers:
            stats = metrics_dict[l]
            theory_medians.append(stats["theory_ratio"]["median"])
            jvp_medians.append(stats["jvp_ratio"]["median"])
            emp_medians.append(stats["emp_ratio"]["median"])
            jvp_p10.append(stats["jvp_ratio"]["p10"])
            jvp_p90.append(stats["jvp_ratio"]["p90"])
            
        plt.figure(figsize=(10, 6))
        
        # Plot theoretical 1/S (or scale/S)
        plt.plot(layers_arr, theory_medians, marker="^", color="crimson", linewidth=2.5, 
                 label=r"Analytical Theory $1/S$")
                 
        # Plot JVP ratio
        plt.plot(layers_arr, jvp_medians, marker="o", color="navy", linewidth=2, linestyle="--", 
                 label="JVP Scaling Ratio")
        plt.fill_between(layers_arr, jvp_p10, jvp_p90, color="navy", alpha=0.15)
        
        # Plot Empirical ratio
        plt.plot(layers_arr, emp_medians, marker="s", color="darkorange", linewidth=1.5, linestyle=":", 
                 label=f"Empirical Scaling Ratio ($\epsilon$={target_radius:.1e})")
                 
        plt.xlabel("Layer Index")
        plt.ylabel(r"Scaling Gain $\|\delta y\|_2 / \|\delta x_{\perp}\|_2$")
        plt.grid(True, alpha=0.3)
        plt.legend(loc="best")
        plt.tight_layout()
        
        plot_name = f"normalization_scaling_verification_{norm_name}.png"
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
            
            cos_sim_medians = []
            cos_sim_p10 = []
            cos_sim_p90 = []
            
            for l in sorted_layers:
                stats = metrics_dict[l]
                cos_sim_medians.append(stats["cos_sim_jvp_x"]["median"])
                cos_sim_p10.append(stats["cos_sim_jvp_x"]["p10"])
                cos_sim_p90.append(stats["cos_sim_jvp_x"]["p90"])
                
            plt.plot(layers_arr, cos_sim_medians, marker=marker, color=color, linewidth=2.5, 
                     label=label_name)
            plt.fill_between(layers_arr, cos_sim_p10, cos_sim_p90, color=color, alpha=0.1)
            plotted = True
            
        if not plotted:
            plt.close()
            continue
            
        # Draw target line of zero alignment (perfect orthogonality)
        plt.axhline(0.0, color="black", linestyle="-", linewidth=1.5)
        
        plt.xlabel("Layer Index")
        plt.ylabel(r"Alignment $\cos(\theta) = \frac{x^T \delta y}{\|x\|_2 \|\delta y\|_2}$")
        plt.ylim(-0.2, 0.2) # zoom in on zero alignment
        plt.grid(True, alpha=0.3)
        plt.legend(loc="best")
        plt.tight_layout()
        
        plot_name = f"normalization_orthogonal_annihilation_{norm_name}.png"
        plt.savefig(os.path.join(plots_dir, plot_name), dpi=dpi)
        plt.close()
        print(f"Saved orthogonal annihilation verification to {os.path.join(plots_dir, plot_name)}")

if __name__ == "__main__":
    main()
