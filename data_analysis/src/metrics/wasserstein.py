
import numpy as np
import torch
from scipy.stats import wasserstein_distance
from typing import Tuple, Optional, Dict
from config import CONFIG
import os
from src.viz.plots import plot_time_series_for_pair

def compare_trajectories(
    a: np.ndarray,
    b: np.ndarray,
    *,
    return_timeseries: bool = True,
    **kwargs,
) -> Tuple[Optional[np.ndarray], Dict[str, float]]:
    if isinstance(a, torch.Tensor):
        a = a.cpu().numpy()
    if isinstance(b, torch.Tensor):
        b = b.cpu().numpy()

    window_size = CONFIG["sliding_window"]["window_size"]
    displacement = CONFIG["sliding_window"]["displacement"]
    use_window = CONFIG["sliding_window"]["use_window"]

    if not use_window:
        # If not using a sliding window, compute the distance over the whole trajectories
        # Note: wasserstein_distance is 1D, so we might need to decide how to handle (T, D) trajectories.
        # A simple approach is to flatten the trajectories, but this loses temporal structure.
        # A better approach might be to compute it per-dimension and average, or on the distribution of values.
        # Here, we'll compute it on the flattened distribution of values.
        dist = wasserstein_distance(a.flatten(), b.flatten())
        aggregates = {"mean": dist, "median": dist, "std": 0.0, "wasserstein": dist}
        return None, aggregates

    # Sliding window implementation
    min_len = min(len(a), len(b))
    distances = []
    
    for center in range(0, min_len, displacement):
        half = window_size // 2
        s = max(0, center - half)
        e = min(min_len, center + half + (window_size % 2))
        window_a = a[s:e].flatten()
        window_b = b[s:e].flatten()
        if window_a.size == 0 or window_b.size == 0:
            continue
        dist = wasserstein_distance(window_a, window_b)
        # Normalize by window size to account for varying window sizes at edges
        actual_size = e - s
        normalized_dist = dist / actual_size
        distances.append(normalized_dist)

    if not distances:
        return None, {"mean": np.nan, "median": np.nan, "std": np.nan}

    timeseries = np.array(distances)
    aggregates = {
        "mean": np.nanmean(timeseries),
        "median": np.nanmedian(timeseries),
        "std": np.nanstd(timeseries),
    }

    out_root = kwargs.get("out_root", None)
    pair_id = kwargs.get("pair_id", None)
    if out_root and timeseries is not None:
        # Save timeseries plot
        if CONFIG["metrics"].get("save_plots", True):
            os.makedirs(out_root, exist_ok=True)
            plot_fname = os.path.join(out_root, f"wasserstein_timeseries_{pair_id}.png")
            plot_time_series_for_pair(timeseries, plot_fname, title=f"Wasserstein Distance ({pair_id})", ylabel="Wasserstein Distance")
        # Save timeseries array if enabled
        if CONFIG["metrics"].get("save_timeseries_array", False):
            try:
                import numpy as np
                fname = os.path.join(out_root, f"wasserstein_timeseries_{pair_id}.npy")
                np.save(fname, timeseries)
            except Exception:
                pass

    return timeseries if return_timeseries else None, aggregates
