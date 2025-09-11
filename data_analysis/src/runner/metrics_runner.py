# STATUS: PARTIAL
import os
import json
from typing import Dict
import numpy as np
import torch

from config import CONFIG
from src.runner.pair_manager import get_pairs
from src.utils.parallel import map_pairs
from src.io.saver import save_tensor
from src.metrics import cos, dtw_fast, hausdorff, frechet, cross_cos, rank_eigen, cross_corr, cos_sim, wasserstein

METRIC_FUNCTIONS = {
    "cos": cos.compare_trajectories,
    "cos_sim": cos_sim.compare_trajectories,
    "dtw_fast": dtw_fast.compare_trajectories,
    "hausdorff": hausdorff.compare_trajectories,
    "frechet": frechet.compare_trajectories,
    "cross_cos": cross_cos.compare_trajectories,
    "rank_eigen": rank_eigen.compare_trajectories,
    "cross_corr": cross_corr.compare_trajectories,
    "wasserstein": wasserstein.compare_trajectories,
}

def _compute_metric_for_pair(pair, trajectories, metrics_to_run, results_dir):
    i, j = pair
    traj_a = trajectories[i]
    traj_b = trajectories[j]

    # Convert torch Tensors to numpy arrays for metric functions that expect numpy
    try:
        if isinstance(traj_a, torch.Tensor):
            traj_a = traj_a.cpu().numpy()
    except Exception:
        pass
    try:
        if isinstance(traj_b, torch.Tensor):
            traj_b = traj_b.cpu().numpy()
    except Exception:
        pass
    pair_results = {}
    pair_id = f"{i}_{j}"

    # If a sweep is active (set by sweep_analysis), include its info in filenames
    sw = CONFIG.get("sweep", {})
    sweep_param = sw.get("param")
    sweep_value = sw.get("value")
    sweep_suffix = f"_{sweep_param}_{sweep_value}" if sweep_param is not None and sweep_value is not None else ""
    pair_id_sweep = f"{pair_id}{sweep_suffix}"

    for metric_name in metrics_to_run:
        if metric_name in METRIC_FUNCTIONS:
            if CONFIG["logging"]["enabled"]:
                print(f"[metrics_runner] computing metric '{metric_name}' for pair {pair}")
            try:
                # Decide plots directory: for sweeps we want results_root/<metric_name>/
                if os.path.basename(results_dir) == "sweep":
                    plots_dir = os.path.join(os.path.dirname(results_dir), metric_name)
                else:
                    plots_dir = os.path.join(results_dir, "plots")
                os.makedirs(plots_dir, exist_ok=True)

                # Pass pair-specific identifiers and output root so metrics can save per-pair plots
                timeseries, aggregates = METRIC_FUNCTIONS[metric_name](traj_a, traj_b, pair_id=pair_id_sweep, out_root=plots_dir)

                # Normalize scalar aggregates to standard keys so summaries/plots work uniformly
                if not any(k in aggregates for k in ("mean", "median", "std")):
                    # try to find a single numeric value in aggregates
                    numeric_vals = [v for v in aggregates.values() if isinstance(v, (int, float, np.floating, np.integer))]
                    if len(numeric_vals) == 1:
                        val = float(numeric_vals[0])
                        aggregates = {"mean": val, "median": val, "std": 0.0, **aggregates}
                        if CONFIG["logging"]["enabled"]:
                            print(f"[metrics_runner] Normalized scalar aggregates for metric '{metric_name}' on pair {pair} -> mean/median/std")
                    else:
                        # no numeric scalar found; leave as-is but warn
                        if CONFIG["logging"]["enabled"]:
                            print(f"[metrics_runner] Warning: metric '{metric_name}' returned aggregates without mean/median/std for pair {pair}: keys={list(aggregates.keys())}")

                pair_results[metric_name] = aggregates
                if CONFIG["logging"]["enabled"]:
                    print(f"[metrics_runner] computed '{metric_name}' for pair {pair}; aggregates keys={list(aggregates.keys())}")

                # Save timeseries if configured
                if timeseries is not None and CONFIG["pairwise"]["save_all_pair_timeseries"]:
                    ts_dir = os.path.join(results_dir, "timeseries")
                    os.makedirs(ts_dir, exist_ok=True)
                    ts_filename = f"{i}_{j}_{metric_name}.pt"
                    save_tensor(timeseries, os.path.join(ts_dir, ts_filename))
            except Exception as e:
                if CONFIG["logging"]["enabled"]:
                    print(f"Could not compute metric {metric_name} for pair {pair}: {e}")
                import traceback
                traceback.print_exc()
                pair_results[metric_name] = {"error": str(e)}
        else:
            if CONFIG["logging"]["enabled"]:
                print(f"Unknown metric: {metric_name}")

    return (pair, pair_results)

def run_metrics(trajectories: np.ndarray, run_id: str) -> Dict:
    """Computes all configured metrics for all configured pairs of trajectories."""
    results_dir = os.path.join(CONFIG["results_root"], run_id)
    os.makedirs(results_dir, exist_ok=True)
    # Determine which pairs to compute based on pairwise config.
    # Precedence: compute_all_pairs -> reference_index -> explicit pairs_to_plot
    pairwise_cfg = CONFIG.get("pairwise", {})
    cfg_metrics = CONFIG.get("metrics", {})
    declared = cfg_metrics.get("available", [])
    metrics_to_run = []
    # Select metrics that both declared in CONFIG and have implementations in METRIC_FUNCTIONS.
    for metric_name in METRIC_FUNCTIONS.keys():
        # Only consider metrics the user declared in CONFIG['metrics']['available']
        if metric_name not in declared:
            continue

        # Look up config for this metric using name or base alias
        base = metric_name.split('_')[0]
        m_cfg = cfg_metrics.get(metric_name, None) or cfg_metrics.get(base, None) or CONFIG.get(metric_name, None) or CONFIG.get(base, None)

        # If we found a dict-like config, respect its 'enabled' flag (default True)
        if isinstance(m_cfg, dict):
            if m_cfg.get("enabled", True):
                metrics_to_run.append(metric_name)
        else:
            # If no config object was found anywhere, include the metric by default.
            if m_cfg is None or bool(m_cfg):
                metrics_to_run.append(metric_name)

    # Warn about declared metrics that have no implementation
    for m in declared:
        if m not in METRIC_FUNCTIONS:
            if CONFIG["logging"]["enabled"]:
                print(f"[metrics_runner] Warning: metric '{m}' declared in CONFIG but has no implementation and will be skipped.")

    # Build pairs list
    n_trajectories = trajectories.shape[0]
    if pairwise_cfg.get("compute_all_pairs", False):
        pairs = get_pairs(n_trajectories, "all")
        pairing_mode = "all"
    else:
        ref_idx = pairwise_cfg.get("reference_index", None)
        if ref_idx is None:
            # Use explicit list provided in config (pairs_to_plot)
            pairs = pairwise_cfg.get("pairs_to_plot", [])
            pairing_mode = "custom"
        else:
            # Pair the reference index with all others
            pairs = [(ref_idx, i) for i in range(n_trajectories) if i != ref_idx]
            pairing_mode = f"ref{ref_idx}"

    # DEBUG: report selection
    if CONFIG["logging"]["enabled"]:
        print(f"[metrics_runner] pairing_mode={pairing_mode}, declared={declared}")
        print(f"[metrics_runner] metrics_to_run={metrics_to_run}")

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

    if CONFIG["logging"]["enabled"]:
        print(f"Completed metrics: {len(pairs)} pairs computed.")
    return final_results


def main(run_id: str, reduced_tensor: torch.Tensor):
    """Main function to run metrics on a reduced tensor."""
    if CONFIG["logging"]["enabled"]:
        print(f"Starting metrics computation for run_id: {run_id}")
    try:
        results = run_metrics(reduced_tensor, run_id)
        if CONFIG["logging"]["enabled"]:
            print("Metrics computation complete.")
        return results
    except Exception as e:
        if CONFIG["logging"]["enabled"]:
            print(f"An error occurred during metrics computation: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Example usage:
    # python -m src.runner.metrics_runner <run_id> <path_to_reduced_tensor>
    import sys
    if len(sys.argv) != 3:
        print("Usage: python -m src.runner.metrics_runner <run_id> <path_to_reduced_tensor>")
        sys.exit(1)

    run_id_arg = sys.argv[1]
    tensor_path_arg = sys.argv[2]

    # Load the tensor
    try:
        tensor_to_process = torch.load(tensor_path_arg)
        if CONFIG["logging"]["enabled"]:
            print(f"Loaded tensor from {tensor_path_arg} with shape: {tensor_to_process.shape}")
    except Exception as e:
        if CONFIG["logging"]["enabled"]:
            print(f"Failed to load tensor from {tensor_path_arg}: {e}")
        sys.exit(1)

    main(run_id_arg, tensor_to_process)