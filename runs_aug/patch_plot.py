import os

file_path = "/home/grego/Documents/BME/Thesis/TDK_LLM/runs_aug/plot_perturbations.py"

with open(file_path, "r") as f:
    code = f.read()

# 1. Locate the last plt.close() in plot_perturbations (inside the rainbow plotting block)
# We will insert the call to plot_extracted_jacobians right after the block ends (back at 4 spaces indentation).
target_part = """                                plt.savefig(os.path.join(plots_dir, f"{info_key}_scaling_rainbow_{current_metric}_xscale-{x_scale}_yscale-{y_scale}.png"), dpi=dpi)
                                plt.close()

def main():"""

replacement_part = """                                plt.savefig(os.path.join(plots_dir, f"{info_key}_scaling_rainbow_{current_metric}_xscale-{x_scale}_yscale-{y_scale}.png"), dpi=dpi)
                                plt.close()

    # Extract and plot cumulative and layer Jacobians from linear perturbation data
    plot_extracted_jacobians(data, group_titles, plots_dir, plotting_cfg)

def plot_extracted_jacobians(data, group_titles, plots_dir, plotting_cfg):
    \"\"\"
    Extracts the cumulative Jacobian A(l) and the layer-by-layer Jacobian J(l -> l+1)
    from the linear regime of the perturbation propagation data, and plots them.
    \"\"\"
    plot_cum = plotting_cfg.get("plot_extracted_cumulative_jacobian", False)
    plot_lay = plotting_cfg.get("plot_extracted_layer_jacobian", False)
    plot_tog = plotting_cfg.get("plot_extracted_jacobians_together", False)
    
    if not (plot_cum or plot_lay or plot_tog):
        return
        
    metric_list = plotting_cfg.get("metric", ["mean"])
    if isinstance(metric_list, str):
        metric_list = [metric_list]
        
    x_scales = plotting_cfg.get("x_scales", ["linear"])
    y_scales = plotting_cfg.get("y_scales", ["log"])
    if isinstance(x_scales, str):
        x_scales = [x_scales]
    if isinstance(y_scales, str):
        y_scales = [y_scales]
        
    plot_prompt_together = plotting_cfg.get("plot_prompt_together", False)
    plot_prompt_separated = plotting_cfg.get("plot_prompt_separated", True)
    dpi = plotting_cfg.get("dpi", 300)
    
    for group_key, radii_data in data.items():
        if group_key.endswith("_aggregated") and not plot_prompt_together:
            continue
        if not group_key.endswith("_aggregated") and not plot_prompt_separated:
            continue
            
        sorted_radii = sorted(radii_data.keys())
        if not sorted_radii:
            continue
            
        info_key = get_safe_filename_info(group_key, group_titles)
        
        # Identify the linear regime: smallest radii where perturbation is linear
        linear_radii = [r for r in sorted_radii if r <= 0.001]
        if not linear_radii:
            # Fallback to the first few radii if all are > 0.001
            linear_radii = sorted_radii[:3] if len(sorted_radii) >= 3 else sorted_radii
            
        for current_metric in metric_list:
            # Extract layer array from the first radius
            first_radius = sorted_radii[0]
            if current_metric not in radii_data[first_radius]:
                continue
            layer_arr = np.array(radii_data[first_radius]["layers"])
            num_layers = len(layer_arr)
            
            # Compute cumulative gain A(l) for each layer index
            A = np.zeros(num_layers)
            for i in range(num_layers):
                ratios = []
                for r in linear_radii:
                    if current_metric in radii_data[r] and len(radii_data[r][current_metric]) > i:
                        d_val = radii_data[r][current_metric][i]
                        ratios.append(d_val / r)
                if ratios:
                    A[i] = np.mean(ratios)
                else:
                    A[i] = 1.0 # fallback
                    
            # Compute layer-by-layer step gain J(l -> l+1)
            layer_jac = np.zeros(num_layers - 1)
            for i in range(num_layers - 1):
                layer_jac[i] = A[i+1] / np.maximum(A[i], 1e-12)
                
            # Now plot for each combination of scale
            for x_scale in x_scales:
                for y_scale in y_scales:
                    # 1. Plot Cumulative Aggregate Jacobian
                    if plot_cum:
                        plt.figure(figsize=(8, 6))
                        plt.plot(layer_arr, A, marker='o', color='b', linewidth=2, markersize=4)
                        plt.xlabel("Layer")
                        plt.ylabel("Aggregate Jacobian")
                        plt.xscale(x_scale)
                        plt.yscale(y_scale)
                        plt.grid(True, alpha=0.3, which="both")
                        if plotting_cfg.get("show_title", False):
                            plt.title(f"Extracted Aggregate Jacobian | {group_titles[group_key]}")
                        plt.tight_layout()
                        filename = f"{info_key}_extracted_cumulative_jacobian_{current_metric}_xscale-{x_scale}_yscale-{y_scale}.png"
                        plt.savefig(os.path.join(plots_dir, filename), dpi=dpi)
                        plt.close()
                        
                    # 2. Plot Layer Jacobian
                    if plot_lay:
                        plt.figure(figsize=(8, 6))
                        plt.plot(layer_arr[1:], layer_jac, marker='s', color='r', linewidth=2, markersize=4)
                        plt.xlabel("Layer")
                        plt.ylabel("Layer Jacobian")
                        plt.xscale(x_scale)
                        plt.yscale(y_scale)
                        plt.grid(True, alpha=0.3, which="both")
                        if plotting_cfg.get("show_title", False):
                            plt.title(f"Extracted Layer Jacobian | {group_titles[group_key]}")
                        plt.tight_layout()
                        filename = f"{info_key}_extracted_layer_jacobian_{current_metric}_xscale-{x_scale}_yscale-{y_scale}.png"
                        plt.savefig(os.path.join(plots_dir, filename), dpi=dpi)
                        plt.close()
                        
                    # 3. Plot both together
                    if plot_tog:
                        plt.figure(figsize=(8, 6))
                        plt.plot(layer_arr, A, marker='o', color='b', linewidth=2, markersize=4, label="Aggregate Jacobian")
                        plt.plot(layer_arr[1:], layer_jac, marker='s', color='r', linewidth=2, markersize=4, label="Layer Jacobian")
                        plt.xlabel("Layer")
                        plt.ylabel("Jacobian Gain")
                        plt.xscale(x_scale)
                        plt.yscale(y_scale)
                        plt.grid(True, alpha=0.3, which="both")
                        plt.legend(loc="best")
                        if plotting_cfg.get("show_title", False):
                            plt.title(f"Extracted Jacobians | {group_titles[group_key]}")
                        plt.tight_layout()
                        filename = f"{info_key}_extracted_jacobians_together_{current_metric}_xscale-{x_scale}_yscale-{y_scale}.png"
                        plt.savefig(os.path.join(plots_dir, filename), dpi=dpi)
                        plt.close()

def main():"""

# 2. Also update main to load overrides from jacobian_config.yaml
main_target = """def main():
    config_path = "config.yaml"
    if not os.path.exists(config_path):
        print(f"Error: {config_path} not found.")
        return
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    results_dir = config.get("experiment", {}).get("results_dir", "./results_perturbations")
    plotting_cfg = config.get("plotting", {})"""

main_replacement = """def main():
    config_path = "config.yaml"
    if not os.path.exists(config_path):
        print(f"Error: {config_path} not found.")
        return
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    results_dir = config.get("experiment", {}).get("results_dir", "./results_perturbations")
    plotting_cfg = config.get("plotting", {})
    
    # Load overrides from jacobian_config.yaml if it exists
    if os.path.exists("jacobian_config.yaml"):
        with open("jacobian_config.yaml", "r") as f_jac:
            jac_config = yaml.safe_load(f_jac)
            if "plotting" in jac_config:
                for k, v in jac_config["plotting"].items():
                    plotting_cfg[k] = v"""

if target_part in code:
    code = code.replace(target_part, replacement_part)
    print("Replaced target part successfully.")
else:
    # Try with unix newlines explicitly
    target_part_unix = target_part.replace("\r\n", "\n")
    if target_part_unix in code:
        code = code.replace(target_part_unix, replacement_part)
        print("Replaced target part (unix newlines) successfully.")
    else:
        print("Error: target part not found in code!")

if main_target in code:
    code = code.replace(main_target, main_replacement)
    print("Replaced main part successfully.")
else:
    main_target_unix = main_target.replace("\r\n", "\n")
    if main_target_unix in code:
        code = code.replace(main_target_unix, main_replacement)
        print("Replaced main part (unix newlines) successfully.")
    else:
        print("Error: main part not found in code!")

with open(file_path, "w") as f:
    f.write(code)
print("Saved modified file.")
