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
    print("fastdtw not available, falling back to tslearn or slower implementations if present.")

try:
    from tslearn.metrics import dtw as tslearn_dtw
    TSLEARN_AVAILABLE = True
except ImportError:
    TSLEARN_AVAILABLE = False
    print("tslearn not available; DTW exact mode will be unavailable.")

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
        for center in range(0, min_len, displacement):
            s = max(0, center - window_size)
            e = min(min_len, center + window_size)
            seg_a = a[s:e]
            seg_b = b[s:e]
            if seg_a.shape[0] == 0 or seg_b.shape[0] == 0:
                continue
            d, path = _dtw_with_path(seg_a, seg_b)
            len_seg = seg_a.shape[0]
            d_norm = d / len_seg if len_seg > 0 else 0
            distances.append(d_norm)
            positions.append(center)

        aggregates = {"mean": float(np.mean(distances)), "median": float(np.median(distances)), "std": float(np.std(distances))}
        timeseries = np.array(distances) if return_timeseries else None

        out_root = kwargs.get("out_root")
        pair_id = kwargs.get("pair_id", "")
        if timeseries is not None and out_root:
            # Save timeseries plot
            if CONFIG["metrics"].get("save_plots", True):
                try:
                    from src.viz.plots import plot_time_series_for_pair
                    os.makedirs(out_root, exist_ok=True)
                    fname = f"dtw_timeseries_{pair_id}"
                    if window_size is not None:
                        fname += f"_window_size_{window_size}"
                    fname += ".png"
                    plot_time_series_for_pair(timeseries, os.path.join(out_root, fname), title=f"DTW distances ({pair_id})", ylabel="DTW Distance", sweep_param_value=window_size)
                except Exception:
                    pass
            # Save timeseries array if enabled
            if CONFIG["metrics"].get("save_timeseries_array", False):
                try:
                    np.save(os.path.join(out_root, f"dtw_timeseries_{pair_id}.npy"), timeseries)
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
                fname = f"dtw_alignment_timeseries_{kwargs.get('pair_id', '')}"
                if window_size is not None:
                    fname += f"_window_size_{window_size}"
                fname += ".png"
                plot_time_series_for_pair(timeseries, os.path.join(kwargs.get("out_root"), fname), title=f"DTW alignment distances ({kwargs.get('pair_id','')})", ylabel="DTW Distance", sweep_param_value=window_size)
            except Exception:
                pass

    return timeseries, aggregates
