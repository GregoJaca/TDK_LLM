# STATUS: PARTIAL
import os
import json
from typing import Dict
import numpy as np

from config import CONFIG
from src.runner.pair_manager import get_pairs
from src.utils.parallel import map_pairs
from src.io.saver import save_tensor
from src.metrics import cos, dtw_fast, hausdorff, frechet, cross_cos

METRIC_FUNCTIONS = {
    "cos": cos.compare_trajectories,
    "dtw_fast": dtw_fast.compare_trajectories,
    "hausdorff": hausdorff.compare_trajectories,
    "frechet": frechet.compare_trajectories,
    "cross_cos": cross_cos.compare_trajectories,
}

def _compute_metric_for_pair(pair, trajectories, metrics_to_run, results_dir):
    i, j = pair
    traj_a = trajectories[i]
    traj_b = trajectories[j]
    pair_results = {}
    pair_id = f"{i}_{j}"
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    for metric_name in metrics_to_run:
        if metric_name in METRIC_FUNCTIONS:
            try:
                # Pass pair-specific identifiers and output root so metrics can save per-pair plots
                timeseries, aggregates = METRIC_FUNCTIONS[metric_name](traj_a, traj_b, pair_id=pair_id, out_root=plots_dir)
                pair_results[metric_name] = aggregates

                # Save timeseries if configured
                if timeseries is not None and CONFIG["pairwise"]["save_all_pair_timeseries"]:
                    ts_dir = os.path.join(results_dir, "timeseries")
                    os.makedirs(ts_dir, exist_ok=True)
                    ts_filename = f"{i}_{j}_{metric_name}.pt"
                    save_tensor(timeseries, os.path.join(ts_dir, ts_filename))
            except Exception as e:
                print(f"Could not compute metric {metric_name} for pair {pair}: {e}")
                pair_results[metric_name] = {"error": str(e)}
        else:
            print(f"Unknown metric: {metric_name}")

    return (pair, pair_results)

def run_metrics(trajectories: np.ndarray, run_id: str) -> Dict:
    """Computes all configured metrics for all configured pairs of trajectories."""
    results_dir = os.path.join(CONFIG["results_root"], run_id)
    os.makedirs(results_dir, exist_ok=True)
    pairing_mode = CONFIG["metrics"]["default_pairing"]
    # Respect per-metric enabled flags in CONFIG
    cfg_metrics = CONFIG.get("metrics", {})
    declared = cfg_metrics.get("available", [])
    metrics_to_run = []
    for m in declared:
        # Look up metric config; support aliases like 'dtw' -> 'dtw_fast'
        m_cfg = cfg_metrics.get(m, None)
        if m_cfg is None:
            # try base name (before underscore) as alias
            base = m.split('_')[0]
            m_cfg = cfg_metrics.get(base, None)

        if isinstance(m_cfg, dict):
            if m_cfg.get("enabled", True):
                metrics_to_run.append(m)
        else:
            # If no per-metric config, include by default
            metrics_to_run.append(m)

    pairs = get_pairs(trajectories.shape[0], pairing_mode)
    
    # Use a lambda to pass additional arguments to the worker function
    worker_func = lambda pair: _compute_metric_for_pair(pair, trajectories, metrics_to_run, results_dir)

    # This part is currently not working as intended with map_pairs
    # results = map_pairs(worker_func, pairs)
    # For now, run sequentially
    results = [worker_func(p) for p in pairs]

    # Process results
    final_results = {
        "run_id": run_id,
        "config": {
            "pairing_mode": pairing_mode,
            "metrics": metrics_to_run
        },
        "pairs": {f"{p[0]}_{p[1]}": res for p, res in results}
    }

    # Save metrics.json (optional)
    if CONFIG.get("save_metrics_json", True):
        with open(os.path.join(results_dir, "metrics.json"), "w") as f:
            json.dump(final_results, f, indent=4)

    print(f"Completed metrics: {len(pairs)} pairs computed.")
    return final_results
