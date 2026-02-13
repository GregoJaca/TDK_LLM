
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import cosine_similarity
import os

from config import CONFIG
from src.viz.plots import plot_time_series_for_pair

def _calculate_cos_sim_matrix(traj):
    """Calculates the cosine similarity matrix for a single trajectory."""
    return cosine_similarity(traj)

def compare_trajectories(a, b, *, return_timeseries=True, pair_id=None, out_root=None, **kwargs):
    """
    Calculates the cross-correlation of cosine similarity matrices between two trajectories.
    """
    if isinstance(a, np.ndarray):
        a = a
    if isinstance(b, np.ndarray):
        b = b

    cos_sim_a = _calculate_cos_sim_matrix(a)
    cos_sim_b = _calculate_cos_sim_matrix(b)

    window_size = CONFIG["sliding_window"]["window_size"]
    displacement = CONFIG["sliding_window"]["displacement"]
    correlation_type = CONFIG["metrics"]["cross_corr"]["correlation_type"]

    if not CONFIG["sliding_window"]["use_window"]:
        # Perform cross-correlation on the full matrices
        if np.std(cos_sim_a.flatten()) == 0 or np.std(cos_sim_b.flatten()) == 0:
            corr = 0.0
        else:
            if correlation_type == 'pearson':
                corr, _ = pearsonr(cos_sim_a.flatten(), cos_sim_b.flatten())
            elif correlation_type == 'spearman':
                corr, _ = spearmanr(cos_sim_a.flatten(), cos_sim_b.flatten())
            else:
                raise ValueError(f"Unknown correlation type: {correlation_type}")
        
        aggregates = {"mean": corr, "median": corr, "std": 0}
        return None, aggregates

    # Sliding window approach
    min_len = min(len(cos_sim_a), len(cos_sim_b))
    time_series = []
    
    for center in range(0, min_len, displacement):
        s = max(0, center - window_size)
        e = min(min_len, center + window_size)
        window_a = cos_sim_a[s:e, s:e]
        window_b = cos_sim_b[s:e, s:e]

        window_a_flat = window_a.flatten()
        window_b_flat = window_b.flatten()

        if np.std(window_a_flat) == 0 or np.std(window_b_flat) == 0:
            corr = 0.0
        else:
            if correlation_type == 'pearson':
                corr, _ = pearsonr(window_a_flat, window_b_flat)
            elif correlation_type == 'spearman':
                corr, _ = spearmanr(window_a_flat, window_b_flat)
            else:
                # Should be caught by config validation, but as a safeguard:
                raise ValueError(f"Unknown correlation type: {correlation_type}")

        time_series.append(corr)
    
    save_timeseries_array = CONFIG["metrics"].get("save_timeseries_array", False)
    time_series = np.array(time_series) if (return_timeseries or save_timeseries_array) else None
    aggregates = {
        "mean": np.mean(time_series) if time_series is not None else float('nan'),
        "median": np.median(time_series) if time_series is not None else float('nan'),
        "std": np.std(time_series) if time_series is not None else float('nan')
    }

    if time_series is not None and out_root:
        # Save plot
        if CONFIG["metrics"].get("save_plots", True):
            fname = f"cross_corr_{correlation_type}_{pair_id}"
            if CONFIG["sliding_window"]["use_window"]:
                fname += f"_window_size_{window_size}"
            fname += ".png"
            out_path = os.path.join(out_root, fname)
            try:
                # Plot distance-like measure: 1 - correlation
                plot_series = 1.0 - time_series
                plot_time_series_for_pair(
                    plot_series,
                    out_path,
                    title=f"1 - Cross-Correlation ({correlation_type.capitalize()}) for Pair {pair_id}",
                    ylabel=f"1 - Cross-Corr ({correlation_type})",
                    sweep_param_value=window_size
                )
            except Exception:
                # fallback to original plot if something goes wrong
                plot_time_series_for_pair(
                    time_series,
                    out_path,
                    title=f"Cross-Correlation ({correlation_type.capitalize()}) for Pair {pair_id}",
                    ylabel=f"Cross-Corr ({correlation_type})",
                    sweep_param_value=window_size
                )
        # Save timeseries array if enabled
        if CONFIG["metrics"].get("save_timeseries_array", False):
            try:
                fname = os.path.join(out_root, f"cross_corr_timeseries_{correlation_type}_{pair_id}.npy")
                np.save(fname, time_series)
            except Exception:
                pass

    return time_series, aggregates
