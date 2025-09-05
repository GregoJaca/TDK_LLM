# STATUS: PARTIAL
import numpy as np
from typing import Tuple, Optional, Dict
from config import CONFIG
import torch

_EPS = 1e-12
import os
from src.viz.plots import plot_time_series_for_pair

def compare_trajectories(
    a: np.ndarray, 
    b: np.ndarray, 
    *, 
    return_timeseries: bool = True, 
    **kwargs
) -> Tuple[Optional[np.ndarray], Dict[str, float]]:
    """Computes cosine distance between two trajectories, with optional shifts."""
    shifts = kwargs.get("shifts", CONFIG["metrics"]["cos"]["shifts"])
    
    # Accept torch tensors by converting to numpy
    if isinstance(a, torch.Tensor):
        a = a.cpu().numpy()
    if isinstance(b, torch.Tensor):
        b = b.cpu().numpy()

    # Ensure trajectories are float arrays
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    # Compute norms and guard zeros to avoid invalid value warnings
    na = np.linalg.norm(a, axis=-1, keepdims=True)
    nb = np.linalg.norm(b, axis=-1, keepdims=True)
    na[na == 0] = _EPS
    nb[nb == 0] = _EPS

    a = a / na
    b = b / nb

    all_distances = []
    for shift in shifts:
        if shift == 0:
            # No shift
            dist = 1 - np.sum(a * b, axis=-1)
            all_distances.append(dist)
            continue

        # Apply shift
        if shift > 0:
            shifted_a = a[:-shift]
            shifted_b = b[shift:]
        else: # shift < 0
            shifted_a = a[-shift:]
            shifted_b = b[:shift]
        
        dist = 1 - np.sum(shifted_a * shifted_b, axis=-1)
        
        # Pad to original length for consistent aggregation
        padded_dist = np.full(a.shape[0], np.nan)
        if shift > 0:
            padded_dist[shift:] = dist
        else:
            padded_dist[:len(dist)] = dist
        all_distances.append(padded_dist)

    # Aggregate across shifts (e.g., take the minimum distance at each timestep)
    aggregation_method = CONFIG["metrics"]["cos"].get("shift_aggregation", "min")
    if aggregation_method == "mean":
        final_distances = np.nanmean(np.array(all_distances), axis=0)
    else: # Default to min
        final_distances = np.nanmin(np.array(all_distances), axis=0)

    aggregates = {
        "mean": np.nanmean(final_distances),
        "median": np.nanmedian(final_distances),
        "std": np.nanstd(final_distances)
    }

    timeseries = final_distances if return_timeseries else None

    # Optionally save a timeseries plot when an output root is provided
    out_root = kwargs.get("out_root", None)
    pair_id = kwargs.get("pair_id", None)
    if out_root is not None and timeseries is not None and CONFIG["metrics"].get("save_plots", True):
        try:
            os.makedirs(out_root, exist_ok=True)
            plot_fname = os.path.join(out_root, f"cos_timeseries_{pair_id}.png" if pair_id else "cos_timeseries.png")
            plot_time_series_for_pair(timeseries, plot_fname)
        except Exception:
            # non-fatal if plotting fails
            pass
    return timeseries, aggregates
