# STATUS: PARTIAL
import os
import itertools
import pandas as pd
import torch
import numpy as np

from config import CONFIG
from src.io.loader import load_tensor
from src.io.saver import save_tensor
from src.reduce.pca import PCAReducer
from src.runner.metrics_runner import run_metrics
from src.runner.lyapunov import estimate_lyapunov
from src.utils.logging import get_logger

logger = get_logger(__name__)

def make_sweep_id(params):
    return "-_-".join([f"{k}{v}" for k, v in params.items()])

def run_sweep(input_path=None):
    sweep_configs = {
        "r": CONFIG["reduction"]["pca"]["r_values"],
        "shift": CONFIG["metrics"]["cos"]["shifts"],
    }
    param_keys = list(sweep_configs.keys())
    param_combinations = list(itertools.product(*sweep_configs.values()))

    logger.info(f"Starting sweep with {len(param_combinations)} combinations.")

    # Load data once
    input_path = input_path or CONFIG["input_path"]
    if not os.path.exists(input_path):
        logger.warning(f"Input file {input_path} not found. Using example data.")
        input_path = "examples/small_example.pt"
    X = load_tensor(input_path)

    results = []
    for params in param_combinations:
        param_dict = dict(zip(param_keys, params))
        sweep_id = make_sweep_id(param_dict)
        logger.info(f"Running sweep: {param_dict}")

        # --- Reduction (with caching) ---
        cache_dir = "cache/reduction"
        os.makedirs(cache_dir, exist_ok=True)
        reduced_path = os.path.join(cache_dir, f"pca_r{param_dict['r']}.{CONFIG['save']['tensor_save_format']}")

        if os.path.exists(reduced_path):
            logger.info(f"Loading cached reduction from {reduced_path}")
            X_reduced = load_tensor(reduced_path).numpy()
        else:
            logger.info("Performing reduction...")
            reducer = PCAReducer(r=param_dict['r'])
            X_reduced = reducer.fit_transform(X.numpy())
            if CONFIG["save"]["save_reduced_tensors"]:
                save_tensor(torch.from_numpy(X_reduced), reduced_path)
                logger.info(f"Saved reduced tensor to {reduced_path}")
            else:
                logger.info("Skipping saving reduced tensor as per config.")
        
        # --- Metrics & Lyapunov (simplified for sweep) ---
        # In a full implementation, we would pass the sweep params to the runners
        # For now, we just run the default pipeline on the reduced data
        run_id = f"sweep-{sweep_id}"
        metrics_summary = run_metrics(X_reduced, run_id=run_id)
        lyap_summary = estimate_lyapunov(X_reduced, run_id=run_id)

        # Collect results
        sweep_result = param_dict.copy()
        sweep_result["mean_lyap"] = lyap_summary["slope"]
        sweep_result["r2_lyap"] = lyap_summary["r2"]

        # Add more metrics from metrics_summary
        for metric_name in metrics_summary["config"]["metrics"]:
            # Collect all 'mean' values for the current metric across all pairs
            metric_means = []
            for pair_data in metrics_summary["pairs"].values():
                if metric_name in pair_data and "mean" in pair_data[metric_name]:
                    metric_means.append(pair_data[metric_name]["mean"])
            
            if metric_means:
                sweep_result[f"mean_{metric_name}"] = np.mean(metric_means)
                sweep_result[f"std_{metric_name}"] = np.std(metric_means)
            else:
                sweep_result[f"mean_{metric_name}"] = np.nan
                sweep_result[f"std_{metric_name}"] = np.nan

        results.append(sweep_result)

    # --- Save summary --- 
    df = pd.DataFrame(results)
    summary_path = os.path.join(CONFIG["results_root"], "sweep_summary.csv")
    df.to_csv(summary_path, index=False)
    logger.info(f"Sweep complete. Summary saved to {summary_path}")

    # --- Plot sweep results ---
    plot_path_prefix = os.path.join(CONFIG["results_root"], "sweep_plots")
    plot_hyperparam_sweep(df, plot_path_prefix)
    logger.info(f"Sweep plots saved to {plot_path_prefix}_*.png")

if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=None, help="Path to the input tensor file")
    args = parser.parse_args()
    run_sweep(input_path=args.input)
