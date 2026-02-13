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
    print("scipy.spatial.cKDTree not available; falling back to slower nearest-neighbor implementation.")

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
            for center in range(0, min_len, displacement):
                s = max(0, center - window_size)
                e = min(min_len, center + window_size)
                wa = a[s:e]
                wb = b[s:e]
                # compute symmetric per-window Hausdorff via per-point mins
                # GGG why doing this distance manually and not using directed_hausdorff? is this equivalent? (i think not and probably max_of_min would be equivalent) what would be the equivalent to directed_hausdorff?
                a_to_b = per_point_min_dists(wa, wb)
                b_to_a = per_point_min_dists(wb, wa)
                aggregation_method = CONFIG["metrics"]["hausdorff"].get("aggregation", "max_of_mean")
                if aggregation_method == "mean_of_max":
                    val = np.mean([np.max(a_to_b), np.max(b_to_a)])
                else: # Default to max_of_mean
                    val = max(np.mean(a_to_b), np.mean(b_to_a))
                values.append(val)
                centers.append(center)

            save_timeseries_array = CONFIG["metrics"].get("save_timeseries_array", False)
            timeseries = np.array(values) if (return_timeseries or save_timeseries_array) else None
            aggregates = {"mean": float(np.mean(values)), "median": float(np.median(values)), "std": float(np.std(values))}

            # optionally save timeseries plot

            out_root = kwargs.get("out_root")
            pair_id = kwargs.get("pair_id", "")
            if timeseries is not None:
                # Save timeseries plot
                if CONFIG["metrics"].get("save_plots", True) and out_root:
                    try:
                        from src.viz.plots import plot_time_series_for_pair
                        os.makedirs(out_root, exist_ok=True)
                        fname = f"hausdorff_timeseries_{pair_id}"
                        if window_size is not None:
                            fname += f"_window_size_{window_size}"
                        fname += ".png"
                        plot_time_series_for_pair(timeseries, os.path.join(out_root, fname), title=f"Hausdorff distances ({pair_id})", ylabel="Hausdorff Distance", sweep_param_value=window_size)
                    except Exception:
                        pass
                # Save timeseries array if enabled
                if CONFIG["metrics"].get("save_timeseries_array", False) and out_root:
                    try:
                        np.save(os.path.join(out_root, f"hausdorff_timeseries_{pair_id}.npy"), timeseries)
                    except Exception:
                        pass

    return timeseries, aggregates
