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

def main(input_path, results_root, sweep_param_value=None):
    logger.info(f"run_all.main called with input_path={input_path}, results_root={results_root}, sweep_param_value={sweep_param_value}")
    # debug
    embedding_cfg_dbg = CONFIG.get('EMBEDDING_CONFIG', {})
    logger.info(f"EMBEDDING_CONFIG={embedding_cfg_dbg}")
    logger.info(f"CONFIG.current_embed_raw={CONFIG.get('current_embed_raw')}")
    if sweep_param_value is not None:
        # When running as part of a sweep, the caller provides a per-embed
        # results_root. Do not create an extra 'sweep' subdirectory there
        # (it previously produced unwanted '.../sweep' folders).
        run_id = 'sweep'
        results_dir = results_root
    else:
        run_id = make_run_id()
        results_dir = os.path.join(results_root, run_id)
    # Create a top-level plots folder only for non-sweep (regular) runs.
    if sweep_param_value is None:
        plots_dir = os.path.join(results_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
    else:
        plots_dir = None
    logger.info(f"Starting run {run_id}")

    # Determine input loading mode: legacy single-file or per-trajectory files.
    embedding_cfg = globals().get('CONFIG', {}).get('EMBEDDING_CONFIG', {})
    input_mode = embedding_cfg.get('input_mode', 'single_file')

    if input_mode == 'per_trajectory':
        # Expect input_path to point to a single example trajectory file or a template path.
        # If input_path is a file that exists, treat it as a single trajectory; otherwise
        # assume it's a template or path to a .pt and let metrics_runner handle per-trajectory loading.
        # For compatibility, if input_path is a directory, list .pt files inside.
        X = None
        trajectories = []
        if os.path.isdir(input_path):
            files = sorted([os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith('.pt')])
            for p in files:
                try:
                    t = load_tensor(p)
                    trajectories.append(t)
                except Exception:
                    logger.warning(f"Failed to load trajectory file {p}; skipping.")
        else:
            # single file path: try to load; could be a single-trajectory file
            if os.path.exists(input_path):
                try:
                    t = load_tensor(input_path)
                    # if tensor has shape (T,D) treat as single trajectory
                    if hasattr(t, 'ndim') and t.ndim == 2:
                        trajectories.append(t)
                    else:
                        # if it's (n,T,D), flatten into list
                        try:
                            for i in range(t.shape[0]):
                                trajectories.append(t[i])
                        except Exception:
                            trajectories.append(t)
                except Exception:
                    logger.warning(f"Could not load input_path {input_path} as per-trajectory; falling back to example.")
            else:
                logger.warning(f"Input path {input_path} not found for per_trajectory mode; using example data.")

        # If CONFIG specifies which pairs to compute, load only those indices
        pairwise_cfg = CONFIG.get('pairwise', {})
        pairs_to_plot = pairwise_cfg.get('pairs_to_plot', [])
        indices_to_load = None
        if pairwise_cfg.get('compute_all_pairs', False):
            indices_to_load = None
        else:
            if pairwise_cfg.get('reference_index', None) is not None:
                n_try = None
                # leave indices_to_load None to let loader decide later
            else:
                # collect unique indices from explicit pairs_to_plot
                idxs = set()
                for p in pairs_to_plot:
                    try:
                        i0, i1 = int(p[0]), int(p[1])
                        idxs.add(i0)
                        idxs.add(i1)
                    except Exception:
                        pass
                if len(idxs) > 0:
                    indices_to_load = sorted(list(idxs))

        if len(trajectories) == 0:
            logger.warning("No per-trajectory files found; loading example data.")
            X = load_tensor("examples/small_example.pt")
        else:
            # If indices_to_load is set, try to load only those files by building filenames
            current_embed = CONFIG.get('current_embed_raw')
            embed_cfg = CONFIG.get('EMBEDDING_CONFIG', {})
            input_template = embed_cfg.get('input_template', '{embed}.pt')
            base_folder = input_path if os.path.isdir(input_path) else os.path.dirname(input_path)

            if indices_to_load is not None and current_embed is not None:
                trajectories = []
                for i in indices_to_load:
                    fname = input_template.format(embed=current_embed, i=i)
                    p = os.path.join(base_folder, fname)
                    if os.path.exists(p):
                        try:
                            t = load_tensor(p)
                            trajectories.append(t)
                        except Exception:
                            logger.warning(f"Failed to load per-trajectory file {p}; skipping.")
                    else:
                        logger.warning(f"Per-trajectory file {p} not found; skipping.")
                if len(trajectories) == 0:
                    logger.warning("No requested per-trajectory files loaded; falling back to previously loaded trajectories.")

            # Stack trajectories into an (n, T, D) tensor if possible
            try:
                import torch
                X = torch.stack([torch.as_tensor(t) for t in trajectories], dim=0)
            except Exception:
                # fallback: convert to numpy then to tensor
                import numpy as _np
                lst = [(_np.asarray(t)) for t in trajectories]
                X = torch.as_tensor(_np.stack(lst, axis=0))

        logger.info("Loaded per-trajectory input; num trajectories=%d", X.shape[0])
        logger.info("Loaded tensor shape %s", tuple(X.shape))
    else:
        if not os.path.exists(input_path):
            logger.warning(f"Input file {input_path} not found. Using example data.")
            input_path = "examples/small_example.pt"

        X = load_tensor(input_path)
        # Log the input tensor path and shape so it's clear which file was used
        logger.info("Loaded tensor file: %s", input_path)
        logger.info("Loaded tensor shape %s", tuple(X.shape))

    # Reduction
    X_reduced = X
    if CONFIG.get("reduction", {}).get("pca", {}).get("enabled", True):
        reducer = PCAReducer(r=CONFIG["reduction"]["pca"]["r_values"][0])  # example default r=16
        X_reduced = reducer.fit_transform(X.numpy())
        if CONFIG["save"]["save_reduced_tensors"]:
            save_tensor(torch.from_numpy(X_reduced), os.path.join(results_dir, CONFIG["save"]["tensor_names"]["pca"] + "." + CONFIG["save"]["tensor_save_format"]))
            logger.info("Reduction done")
        else:
            logger.info("Reduction done (saving of reduced tensors skipped as per config).")
    else:
        logger.info("PCA reduction is disabled in config; skipping reduction.")

    # Plot PCA explained variance (only if PCA was run). Only generate and
    # save the PCA plot for non-sweep runs (user prefers no top-level plots
    # folder or PCA output during sweeps).
    if sweep_param_value is None and plots_dir is not None:
        pca_plot_path = os.path.join(plots_dir, "pca_explained_variance.png")
    else:
        pca_plot_path = None

    if pca_plot_path is not None and CONFIG.get("reduction", {}).get("pca", {}).get("enabled", True) and 'reducer' in locals():
        plot_pca_explained_variance(reducer.model, pca_plot_path, sweep_param_value=sweep_param_value)
    else:
        # mark as not available
        pca_plot_path = None

    # Metrics
    # run_metrics will internally respect per-metric enabled flags from CONFIG
    metrics_summary = None
    # Run metrics when any per-metric 'enabled' flag is True
    any_enabled = any(
        isinstance(v, dict) and bool(v.get('enabled', False))
        for v in CONFIG.get('metrics', {}).values()
    )
    if any_enabled:
        metrics_summary = run_metrics(X_reduced, run_id=run_id)
        logger.info("Metrics complete")
    else:
        logger.info("No metrics enabled in config; skipping metrics.")

    # Plotting for metrics that were actually computed (prefer metrics_summary).
    metrics_list_for_plots = None
    if metrics_summary and "config" in metrics_summary and "metrics" in metrics_summary["config"]:
        metrics_list_for_plots = metrics_summary["config"]["metrics"]
    else:
        # Fall back to listing metrics that are enabled in CONFIG
        metrics_list_for_plots = [k for k, v in CONFIG.get('metrics', {}).items() if isinstance(v, dict) and v.get('enabled', False)]

    save_hist = CONFIG.get('plots', {}).get('save_histograms', True)
    if save_hist:
        for metric_name in metrics_list_for_plots:
            for agg_type in ["mean", "median", "std"]:
                if sweep_param_value is not None:
                    plot_dir = os.path.join(results_root, metric_name)
                    os.makedirs(plot_dir, exist_ok=True)
                    plot_path = os.path.join(plot_dir, f"pairwise_distance_hist_{metric_name}_{agg_type}_window_size_{sweep_param_value}.png")
                else:
                    # For regular runs, use the top-level plots_dir created earlier.
                    plot_path = os.path.join(plots_dir, f"pairwise_distance_hist_{metric_name}_{agg_type}.png")
                logger.info(f"Saving plot to {plot_path}")
                plot_pairwise_distance_distribution(metrics_summary, plot_path, metric_name=metric_name, aggregate_type=agg_type, sweep_param_value=sweep_param_value)
    else:
        logger.info("Skipping histogram plots: CONFIG['plots']['save_histograms'] is False")

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
        summary_lines.append(f"  Plot: {pca_plot_path if pca_plot_path is not None else 'not_saved_in_sweep'}")
    else:
        summary_lines.append("  PCA not run or not available.")
    summary_lines.append("")
    summary_lines.append("Metrics Summary:")
    # Use the computed metrics list when available for the summary aggregation
    metrics_for_summary = metrics_list_for_plots
    for metric_name in metrics_for_summary:
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
            summary_lines.append(f"    Plot: {os.path.join(plots_dir, f'pairwise_distance_hist_{metric_name}_mean.png') if plots_dir is not None else 'not_saved_in_sweep'}")
    summary_lines.append("")
    summary_lines.append("Lyapunov Estimate:")
    if lyap:
        summary_lines.append(f"  Slope: {np.round(lyap.get('slope', 0), 4)}")
        summary_lines.append(f"  R2: {np.round(lyap.get('r2', 0), 4)}")
        summary_lines.append(f"  Window: {lyap.get('linear_window', [])}")
        summary_lines.append(f"  Plot: {os.path.join(plots_dir, 'mean_log_vs_time.png') if plots_dir is not None else 'not_saved_in_sweep'}")

    # Save summary only for non-sweep runs (user prefers no summary file during sweeps)
    if sweep_param_value is None:
        with open(os.path.join(results_dir, "summary.txt"), "w") as f:
            f.write("\n".join(summary_lines))

    print("Run complete:", run_id)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to the input tensor file")
    parser.add_argument("--results", type=str, default="results", help="Path to the results directory")
    args = parser.parse_args()
    main(input_path=args.input, results_root=args.results)
