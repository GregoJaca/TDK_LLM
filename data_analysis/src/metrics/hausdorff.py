# STATUS: DONE
import numpy as np
from typing import Tuple, Optional, Dict
from scipy.spatial.distance import directed_hausdorff
import os
from config import CONFIG

try:
    from scipy.spatial import cKDTree
    KD_AVAILABLE = True
except Exception:
    KD_AVAILABLE = False

def compare_trajectories(
    a: np.ndarray, 
    b: np.ndarray, 
    *, 
    return_timeseries: bool = True, 
    **kwargs
) -> Tuple[Optional[np.ndarray], Dict[str, float]]:
    """Computes the symmetric Hausdorff distance between two trajectories."""
    d_ab = directed_hausdorff(a, b)[0]
    d_ba = directed_hausdorff(b, a)[0]
    symmetric_d = max(d_ab, d_ba)

    aggregates = {"hausdorff_distance": symmetric_d}

    # Optionally compute per-timepoint nearest-neighbor distances and sliding-window timeseries
    sw = CONFIG.get("sliding_window", {})
    use_window = kwargs.get("use_window", sw.get("use_window", False))
    window_size = kwargs.get("window_size", sw.get("window_size", None))
    displacement = kwargs.get("displacement", sw.get("displacement", 1))

    timeseries = None
    # compute per-point nearest distances (A->B) and optionally windowed aggregation
    def per_point_min_dists(X, Y):
        if KD_AVAILABLE:
            tree = cKDTree(Y)
            dists, _ = tree.query(X)
            return dists
        else:
            return np.array([np.min(np.linalg.norm(Y - x, axis=-1)) for x in X])

    if use_window and window_size is not None:
        min_len = min(a.shape[0], b.shape[0])
        if min_len >= window_size:
            values = []
            centers = []
            for start in range(0, min_len - window_size + 1, displacement):
                wa = a[start : start + window_size]
                wb = b[start : start + window_size]
                # compute symmetric per-window Hausdorff via per-point mins
                a_to_b = per_point_min_dists(wa, wb)
                b_to_a = per_point_min_dists(wb, wa)
                # use maximum of mean distances as a representative
                val = max(np.mean(a_to_b), np.mean(b_to_a))
                values.append(val)
                centers.append(start + window_size // 2)

            timeseries = np.array(values) if return_timeseries else None
            aggregates = {"mean": float(np.mean(values)), "median": float(np.median(values)), "std": float(np.std(values))}

            # optionally save timeseries plot
            if timeseries is not None and CONFIG["metrics"].get("save_plots", True) and kwargs.get("out_root"):
                try:
                    from src.viz.plots import plot_time_series_for_pair
                    os.makedirs(kwargs.get("out_root"), exist_ok=True)
                    plot_time_series_for_pair(timeseries, os.path.join(kwargs.get("out_root"), f"hausdorff_timeseries_{kwargs.get('pair_id','')}.png"))
                except Exception:
                    pass

    return timeseries, aggregates
