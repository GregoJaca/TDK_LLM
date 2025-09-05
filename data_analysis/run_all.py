# STATUS: PARTIAL

import sys
import os
import importlib.util
import importlib

# # Robustly load config.py (located alongside this script) and register it as module 'config'
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# config_path = os.path.join(SCRIPT_DIR, "src/config.py")
# if not os.path.exists(config_path):
#     # try parent directory
#     config_path = os.path.join(os.path.dirname(SCRIPT_DIR), "config.py")

# if not os.path.exists(config_path):
#     raise FileNotFoundError(f"Could not find config.py at {config_path}")

# spec = importlib.util.spec_from_file_location("config", config_path)
# config_mod = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(config_mod)
# # register so normal imports work
# sys.modules["config"] = config_mod
# CONFIG = config_mod.CONFIG

from config import CONFIG

from src.io.loader import load_tensor
from src.io.saver import save_tensor
from src.reduce.pca import PCAReducer
from src.runner.metrics_runner import run_metrics
from src.runner.lyapunov import estimate_lyapunov
from src.viz.plots import plot_pairwise_distance_distribution, plot_pca_explained_variance
from src.utils.logging import get_logger
import time
import random
import torch
import numpy as np

logger = get_logger(__name__)

def make_run_id():
    ts = time.strftime(CONFIG["run_id_format"])
    suffix = hex(random.getrandbits(24))[2:]
    return f"{ts}-{suffix}"

def main(input_path=None):
    run_id = make_run_id()
    results_dir = os.path.join(CONFIG["results_root"], run_id)
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    logger.info(f"Starting run {run_id}")

    input_path = input_path or CONFIG["input_path"]
    if not os.path.exists(input_path):
        logger.warning(f"Input file {input_path} not found. Using example data.")
        input_path = "examples/small_example.pt"

    X = load_tensor(input_path)
    logger.info("Loaded tensor shape %s", tuple(X.shape))

    # Reduction
    X_reduced = X
    if CONFIG.get("reduction", {}).get("pca", {}).get("enabled", True):
        reducer = PCAReducer(r=CONFIG["reduction"]["pca"]["r_values"][0])  # example default r=16
        X_reduced = reducer.fit_transform(X.numpy())
        if CONFIG["save"]["save_reduced_tensors"]:
            save_tensor(torch.from_numpy(X_reduced), os.path.join(results_dir, CONFIG["save"]["tensor_names"]["pca"] + ".pt"))
            logger.info("Reduction done")
        else:
            logger.info("Reduction done (saving of reduced tensors skipped as per config).")
    else:
        logger.info("PCA reduction is disabled in config; skipping reduction.")

    # Plot PCA explained variance (only if PCA was run)
    pca_plot_path = os.path.join(plots_dir, "pca_explained_variance.png")
    if CONFIG.get("reduction", {}).get("pca", {}).get("enabled", True) and 'reducer' in locals():
        plot_pca_explained_variance(reducer.model, pca_plot_path)
    else:
        # mark as not available
        pca_plot_path = None

    # Metrics
    # run_metrics will internally respect per-metric enabled flags from CONFIG
    metrics_summary = None
    if CONFIG.get("metrics", {}).get("available"):
        metrics_summary = run_metrics(X_reduced, run_id=run_id)
        logger.info("Metrics complete")
    else:
        logger.info("No metrics declared in config; skipping metrics.")

    # Plotting for all metrics in config
    for metric_name in CONFIG["metrics"]["available"]:
        for agg_type in ["mean", "median", "std"]:
            plot_path = os.path.join(plots_dir, f"pairwise_distance_hist_{metric_name}_{agg_type}.png")
            plot_pairwise_distance_distribution(metrics_summary, plot_path, metric_name=metric_name, aggregate_type=agg_type)

    logger.info("Plotting done")

    # Lyapunov
    lyap = None
    if CONFIG.get("lyapunov", {}).get("enabled", True):
        lyap = estimate_lyapunov(X_reduced, run_id=run_id)
        logger.info("Lyapunov done")
    else:
        logger.info("Lyapunov computation is disabled in config; skipping.")

    # Generate summary report
    summary_lines = []
    summary_lines.append(f"Run ID: {run_id}")
    summary_lines.append(f"Input file: {input_path}")
    summary_lines.append(f"Tensor shape: {tuple(X.shape)}")
    summary_lines.append("")
    summary_lines.append("PCA Explained Variance:")
    if CONFIG.get("reduction", {}).get("pca", {}).get("enabled", True) and 'reducer' in locals() and "explained_variance_ratio" in getattr(reducer, 'model', {}):
        evr = reducer.model["explained_variance_ratio"]
        summary_lines.append(f"  Individual: {np.round(evr, 4)}")
        summary_lines.append(f"  Cumulative: {np.round(np.cumsum(evr), 4)}")
        summary_lines.append(f"  Plot: {pca_plot_path}")
    else:
        summary_lines.append("  PCA not run or not available.")
    summary_lines.append("")
    summary_lines.append("Metrics Summary:")
    for metric_name in CONFIG["metrics"]["available"]:
        metric_means = []
        metric_medians = []
        metric_stds = []
        for pair_data in metrics_summary["pairs"].values():
            if metric_name in pair_data:
                if "mean" in pair_data[metric_name]:
                    metric_means.append(pair_data[metric_name]["mean"])
                if "median" in pair_data[metric_name]:
                    metric_medians.append(pair_data[metric_name]["median"])
                if "std" in pair_data[metric_name]:
                    metric_stds.append(pair_data[metric_name]["std"])
        if metric_means:
            summary_lines.append(f"  {metric_name}:")
            summary_lines.append(f"    Mean: {np.round(np.mean(metric_means), 4)} ± {np.round(np.std(metric_means), 4)}")
            summary_lines.append(f"    Median: {np.round(np.median(metric_medians), 4)}")
            summary_lines.append(f"    Std: {np.round(np.mean(metric_stds), 4)}")
            summary_lines.append(f"    Plot: {os.path.join(plots_dir, f'pairwise_distance_hist_{metric_name}_mean.png')}")
    summary_lines.append("")
    summary_lines.append("Lyapunov Estimate:")
    if lyap:
        summary_lines.append(f"  Slope: {np.round(lyap.get('slope', 0), 4)}")
        summary_lines.append(f"  R2: {np.round(lyap.get('r2', 0), 4)}")
        summary_lines.append(f"  Window: {lyap.get('linear_window', [])}")
        summary_lines.append(f"  Plot: {os.path.join(plots_dir, 'mean_log_vs_time.png')}")

    # Save summary
    with open(os.path.join(results_dir, "summary.txt"), "w") as f:
        f.write("\n".join(summary_lines))

    print("Run complete:", run_id)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=None, help="Path to the input tensor file")
    args = parser.parse_args()
    main(input_path=args.input)
