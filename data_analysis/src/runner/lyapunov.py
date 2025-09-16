# STATUS: PARTIAL
import numpy as np
import json
import os
import torch
from scipy.stats import linregress

from config import CONFIG
from src.runner.pair_manager import get_pairs
from src.io.saver import save_tensor
from src.viz.plots import plot_mean_log_distance_vs_time

def compute_pairwise_logdist(trajs_reduced: np.ndarray, pairs, metric="euclidean") -> np.ndarray:
    """Computes the mean log distance for given pairs of trajectories."""
    if metric != "euclidean":
        raise NotImplementedError("Only euclidean distance is supported for now.")

    all_log_dists = []
    for i, j in pairs:
        dist = np.linalg.norm(trajs_reduced[i] - trajs_reduced[j], axis=-1)
        # Add epsilon to avoid log(0)
        eps = np.finfo(dist.dtype).eps
        all_log_dists.append(np.log(dist + eps))
    
    return np.mean(all_log_dists, axis=0)

def auto_detect_linear_window(times, mean_log_dist, config) -> tuple:
    """Heuristically detects the best linear fitting window."""
    initial_cutoff = int(len(times) * config["initial_time_cutoff_frac"])
    min_len = config["linear_window"]["min_window_len"]
    r2_threshold = config["linear_window"]["r2_threshold"]

    best_window = (0, min_len)
    best_fit = (-np.inf, 0, 0)  # r2, slope, intercept

    for start in range(initial_cutoff - min_len):
        for end in range(start + min_len, initial_cutoff):
            window_slice = slice(start, end)
            t_window = times[window_slice]
            logd_window = mean_log_dist[window_slice]

            if len(t_window) < 2:
                continue

            slope, intercept, r_value, _, _ = linregress(t_window, logd_window)
            r2 = r_value**2

            # Prioritize longer windows that meet the R2 threshold
            if r2 >= r2_threshold:
                if (end - start) > (best_window[1] - best_window[0]):
                    best_window = (start, end)
                    best_fit = (r2, slope, intercept)
            # If no window meets the threshold, take the one with the best R2
            elif best_fit[0] < 0 and r2 > best_fit[0]:
                best_window = (start, end)
                best_fit = (r2, slope, intercept)

    return best_window[0], best_window[1], best_fit[0], best_fit[1], best_fit[2]

def estimate_lyapunov(trajs_reduced: np.ndarray, run_id: str):
    """Estimates the Lyapunov exponent from trajectories."""
    results_dir = os.path.join(CONFIG["results_root"], run_id)
    os.makedirs(results_dir, exist_ok=True)
    lyapunov_config = CONFIG["lyapunov"]
    pairing_mode = CONFIG["metrics"]["default_pairing"]

    pairs = get_pairs(trajs_reduced.shape[0], pairing_mode)
    mean_log_dist = compute_pairwise_logdist(trajs_reduced, pairs)

    times = np.arange(len(mean_log_dist))
    start_idx, end_idx, r2, slope, intercept = auto_detect_linear_window(times, mean_log_dist, lyapunov_config)

    # Save results. Skip saving JSON and plots when running inside a sweep
    # (user does not want 'plots' or sweep-level summary files).
    lyapunov_results = {
        "slope": slope,
        "intercept": intercept,
        "r2": r2,
        "linear_window": [start_idx, end_idx],
        "mean_log_dist": mean_log_dist.tolist() # for JSON serialization
    }

    sw = CONFIG.get('sweep')
    if not sw:
        with open(os.path.join(results_dir, "lyapunov.json"), "w") as f:
            json.dump(lyapunov_results, f, indent=4)

        # Save raw data for plotting
        if CONFIG["save"]["save_reduced_tensors"]:
            save_tensor(torch.from_numpy(mean_log_dist), os.path.join(results_dir, "raw_for_lyapunov.pt"))

        # Generate plot
        plots_dir = os.path.join(results_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        plot_path = os.path.join(plots_dir, "mean_log_vs_time.png")
        plot_mean_log_distance_vs_time(
            mean_log_dist,
            plot_path,
            window=(start_idx, end_idx),
            slope=slope,
            r2=r2,
        )

    print(f"Lyapunov estimation complete. Slope: {slope:.4f}, R2: {r2:.4f}")
    return lyapunov_results
