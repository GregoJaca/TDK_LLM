# STATUS: PARTIAL
import numpy as np
import os
from config import CONFIG
from typing import Tuple, Optional, Dict
from scipy.spatial.distance import euclidean

try:
    from fastdtw import fastdtw
    FASTDTW_AVAILABLE = True
except ImportError:
    FASTDTW_AVAILABLE = False

try:
    from tslearn.metrics import dtw as tslearn_dtw
    TSLEARN_AVAILABLE = True
except ImportError:
    TSLEARN_AVAILABLE = False

def compare_trajectories(
    a: np.ndarray, 
    b: np.ndarray, 
    *, 
    return_timeseries: bool = True, 
    **kwargs
) -> Tuple[Optional[np.ndarray], Dict[str, float]]:
    """Computes Dynamic Time Warping distance between two trajectories."""
    # Use unified sliding window config if present
    sw = CONFIG.get("sliding_window", {})
    use_window = kwargs.get("use_window", sw.get("use_window", False))
    window_size = kwargs.get("window_size", sw.get("window_size", None))
    displacement = kwargs.get("displacement", sw.get("displacement", 1))

    exact = kwargs.get("exact", False)

    # Helper to compute DTW distance and optional path for two sequences
    def _dtw_with_path(x, y):
        if exact and TSLEARN_AVAILABLE:
            d = tslearn_dtw(x, y)
            return d, None
        elif FASTDTW_AVAILABLE:
            d, path = fastdtw(x, y, radius=1, dist=euclidean)
            return d, path
        else:
            raise ImportError("Neither fastdtw nor tslearn is installed. Please install one to use DTW.")

    if use_window and window_size is not None:
        min_len = min(a.shape[0], b.shape[0])
        if min_len < window_size:
            return None, {"dtw_distance": float('nan')}

        distances = []
        positions = []
        for start in range(0, min_len - window_size + 1, displacement):
            seg_a = a[start : start + window_size]
            seg_b = b[start : start + window_size]
            d, path = _dtw_with_path(seg_a, seg_b)
            distances.append(d)
            positions.append(start + window_size // 2)

        aggregates = {"mean": float(np.mean(distances)), "median": float(np.median(distances)), "std": float(np.std(distances))}
        timeseries = np.array(distances) if return_timeseries else None

        # Save timeseries plot if configured
        if return_timeseries and CONFIG["metrics"].get("save_plots", True) and kwargs.get("out_root"):
            try:
                from src.viz.plots import plot_time_series_for_pair
                os.makedirs(kwargs.get("out_root"), exist_ok=True)
                plot_time_series_for_pair(timeseries, os.path.join(kwargs.get("out_root"), f"dtw_timeseries_{kwargs.get('pair_id', '')}.png"))
            except Exception:
                pass

        return timeseries, aggregates

    # Full-trajectory DTW
    d, path = _dtw_with_path(a, b)
    aggregates = {"dtw_distance": d}

    timeseries = None
    if return_timeseries and path is not None:
        timeseries = np.array([euclidean(a[i], b[j]) for i, j in path])

        if CONFIG.get("plots", {}).get("save_timeseries", False) and kwargs.get("out_root"):
            try:
                from src.viz.plots import plot_time_series_for_pair
                os.makedirs(kwargs.get("out_root"), exist_ok=True)
                plot_time_series_for_pair(timeseries, os.path.join(kwargs.get("out_root"), f"dtw_alignment_timeseries_{kwargs.get('pair_id', '')}.png"))
            except Exception:
                pass

    return timeseries, aggregates
