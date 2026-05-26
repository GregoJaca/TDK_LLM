import os
import re
import json
import glob
import yaml
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

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

def main():
    config_path = "jacobian_config.yaml"
    if os.path.exists(config_path):
        config = load_config(config_path)
    else:
        print("Error: jacobian_config.yaml not found.")
        return
        
    exp_config = config.get("experiment", {})
    mft_config = config.get("mft_validation", {})
    plotting_cfg = config.get("plotting", {})
    
    base_results_dir = exp_config.get("results_dir", "./results_jacobians")
    metrics_filename = mft_config.get("metrics_filename", "mft_metrics.json")
    plot_filename_base = mft_config.get("plot_filename", "mft_validation_plot.png")
    
    y_limit_cv = mft_config.get("y_limit_cv", 0.25)
    ratio_tolerance_low = mft_config.get("ratio_tolerance_low", 0.90)
    ratio_tolerance_high = mft_config.get("ratio_tolerance_high", 1.10)
    
    plots_dir = os.path.join(base_results_dir, "aggregated_plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Apply global styling parameters from plotting block
    dpi = plotting_cfg.get("dpi", 300)
    font_size = plotting_cfg.get("font_size", 14)
    label_size = plotting_cfg.get("label_size", 14)
    tick_size = plotting_cfg.get("tick_size", 12)
    legend_size = plotting_cfg.get("legend_size", 11)
    
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
    
    # Find all run directories containing config.json and the MFT metrics file
    config_files = glob.glob(os.path.join(base_results_dir, "*", "config.json"))
    if not config_files:
        print(f"No run configs found in {base_results_dir}.")
        return
        
    # Group measurements
    # grouped_metrics[group_key][layer_idx] = list of layer metric dicts
    grouped_metrics = defaultdict(lambda: defaultdict(list))
    group_titles = {}
    
    for cfg_path in config_files:
        run_dir = os.path.dirname(cfg_path)
        mft_path = os.path.join(run_dir, metrics_filename)
        if not os.path.exists(mft_path):
            continue
            
        with open(cfg_path, "r") as f:
            run_config = json.load(f)
            
        setup_name = run_config["setup"]["name"]
        prompt_hash = run_config.get("prompt_hash", "unknown")
        prompt_text = run_config.get("prompt_text", "Unknown Prompt")
        
        group_key_sep = f"{setup_name}_{prompt_hash}"
        short_prompt = prompt_text if len(prompt_text) < 40 else prompt_text[:37] + "..."
        group_titles[group_key_sep] = f"Setup: {setup_name} | Prompt: '{short_prompt}'"
        
        group_key_tog = f"{setup_name}_aggregated"
        group_titles[group_key_tog] = f"Setup: {setup_name} (All Prompts Aggregated)"
        
        with open(mft_path, "r") as f:
            mft_data = json.load(f)
            
        for layer_str, metrics in mft_data["layers"].items():
            layer_idx = int(layer_str)
            grouped_metrics[group_key_sep][layer_idx].append(metrics)
            grouped_metrics[group_key_tog][layer_idx].append(metrics)
            
    if not grouped_metrics:
        print(f"No {metrics_filename} files found in run directories of {base_results_dir}. Did you run validate_mft.py first?")
        return
        
    print(f"Plotting MFT Validation for {len(grouped_metrics)} groups...")
    
    # Generate the dual-panel plots for each group key
    for group_key, layer_dict in grouped_metrics.items():
        sorted_layers = sorted(layer_dict.keys())
        
        layers_arr = np.array(sorted_layers)
        
        cv_gate_mean = []
        cv_up_mean = []
        cv_down_mean = []
        
        r_option_b_mean = []
        r_t_p10_mean = []
        r_t_p25_mean = []
        r_t_p75_mean = []
        r_t_p90_mean = []
        r_t_median_mean = []
        
        for l_idx in sorted_layers:
            run_metrics = layer_dict[l_idx]
            
            # Static metrics are identical per run, but average in case of minor floats
            cv_gate_mean.append(np.mean([m["CV_gate"] for m in run_metrics]))
            cv_up_mean.append(np.mean([m["CV_up"] for m in run_metrics]))
            cv_down_mean.append(np.mean([m["CV_down"] for m in run_metrics]))
            
            # Dynamic metrics
            r_option_b_mean.append(np.mean([m["R_option_b"] for m in run_metrics]))
            r_t_p10_mean.append(np.mean([m["R_t_p10"] for m in run_metrics]))
            r_t_p25_mean.append(np.mean([m["R_t_p25"] for m in run_metrics]))
            r_t_p75_mean.append(np.mean([m["R_t_p75"] for m in run_metrics]))
            r_t_p90_mean.append(np.mean([m["R_t_p90"] for m in run_metrics]))
            r_t_median_mean.append(np.mean([m["R_t_median"] for m in run_metrics]))
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # --- Subplot 1 (Assumption 2): CV of Weight Norms ---
        ax1.plot(layers_arr, cv_gate_mean, marker='o', color='teal', label=r'$W_{gate}$', linewidth=2, markersize=4)
        ax1.plot(layers_arr, cv_up_mean, marker='s', color='darkorange', label=r'$W_{up}$', linewidth=2, markersize=4)
        ax1.plot(layers_arr, cv_down_mean, marker='^', color='purple', label=r'$W_{down}$', linewidth=2, markersize=4)
        
        ax1.axhline(y=y_limit_cv, color='red', linestyle='--', linewidth=1.5, label=f'Safety Bound ({y_limit_cv})')
        ax1.set_xlabel("Layer Index")
        ax1.set_ylabel("Coefficient of Variation (CV)")
        ax1.set_title("Assumption 2: Uniform Row/Column Norms")
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="upper left")
        
        # --- Subplot 2 (Assumption 1): Diagonal Ratio R ---
        # Main line represents the sequence-averaged ratio (Option B)
        ax2.plot(layers_arr, r_option_b_mean, marker='o', color='navy', label=r'Diagonal Ratio $R$ (Option B)', linewidth=2, markersize=4)
        
        # Fan shading for token-level errors
        ax2.fill_between(layers_arr, r_t_p10_mean, r_t_p90_mean, color='navy', alpha=0.08, label='10th-90th percentile')
        ax2.fill_between(layers_arr, r_t_p25_mean, r_t_p75_mean, color='navy', alpha=0.15, label='25th-75th percentile')
        
        # Perfect Diagonal Isolation (y = 1.0)
        ax2.axhline(y=1.0, color='black', linestyle='-', linewidth=1.5, label='Perfect Isolation (1.0)')
        
        # Tolerance band [0.90, 1.10]
        ax2.axhspan(ratio_tolerance_low, ratio_tolerance_high, color='green', alpha=0.1, label='Tolerance Band')
        
        ax2.set_xlabel("Layer Index")
        ax2.set_ylabel(r"Stretching Ratio $R = \lambda_{diagonal} / \lambda_{true}$")
        ax2.set_title("Assumption 1: Vanishing Off-Diagonals")
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc="upper left")
        
        if plotting_cfg.get("show_title", False):
            plt.suptitle(group_titles[group_key], y=0.98, fontsize=font_size + 2)
            
        plt.tight_layout()
        
        info_key = get_safe_filename_info(group_key, group_titles)
        output_plot_name = plot_filename_base.replace(".png", f"_{info_key}.png")
        output_plot_path = os.path.join(plots_dir, output_plot_name)
        
        plt.savefig(output_plot_path, dpi=dpi)
        plt.close()
        
        print(f"  Saved plot to {output_plot_path}")

if __name__ == "__main__":
    main()
