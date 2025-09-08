# STATUS: PARTIAL
import numpy as np
from typing import Tuple, Optional, Dict
import os
from config import CONFIG

try:
    import fred.fr_dist
    FRED_AVAILABLE = True
except ImportError:
    FRED_AVAILABLE = False
    print("fred.fr_dist not available; using pure-Python Fréchet implementation as fallback.")

try:
    from frechetdist import frdist
    FRECHETDIST_AVAILABLE = True
except ImportError:
    FRECHETDIST_AVAILABLE = False
    print("frechetdist not available; using pure-Python Fréchet implementation as fallback.")

def _discrete_frechet(a, b):
    """A pure Python implementation of discrete Fréchet distance with backtrace.

    Returns (distance, path) where path is a list of (i, j) pairs along the coupling.
    """
    na, nb = len(a), len(b)
    ca = np.full((na, nb), np.inf)
    # DP compute
    for i in range(na):
        for j in range(nb):
            d = np.linalg.norm(a[i] - b[j])
            if i == 0 and j == 0:
                ca[i, j] = d
            else:
                vals = []
                if i > 0:
                    vals.append(ca[i - 1, j])
                if j > 0:
                    vals.append(ca[i, j - 1])
                if i > 0 and j > 0:
                    vals.append(ca[i - 1, j - 1])
                ca[i, j] = max(min(vals), d)

    # backtrace path from (na-1, nb-1)
    path = []
    i, j = na - 1, nb - 1
    path.append((i, j))
    while i > 0 or j > 0:
        choices = []
        if i > 0 and j > 0:
            choices.append((ca[i - 1, j - 1], i - 1, j - 1))
        if i > 0:
            choices.append((ca[i - 1, j], i - 1, j))
        if j > 0:
            choices.append((ca[i, j - 1], i, j - 1))
        # pick the predecessor with smallest ca value
        vals_sorted = sorted(choices, key=lambda x: x[0])
        if not vals_sorted:
            break
        _, i, j = vals_sorted[0]
        path.append((i, j))
    path.reverse()
    return float(ca[na - 1, nb - 1]), path

def compare_trajectories(
    a: np.ndarray, 
    b: np.ndarray, 
    *, 
    return_timeseries: bool = True, 
    **kwargs
) -> Tuple[Optional[np.ndarray], Dict[str, float]]:
    """Computes the Fréchet distance between two trajectories."""
    sw = CONFIG.get("sliding_window", {})
    use_window = kwargs.get("use_window", sw.get("use_window", False))
    window_size = kwargs.get("window_size", sw.get("window_size", None))
    displacement = kwargs.get("displacement", sw.get("displacement", 1))

    timeseries = None
    if FRED_AVAILABLE and not use_window:
        distance = fred.fr_dist(a, b)
        aggregates = {"frechet_distance": distance}
        return None, aggregates
    elif FRECHETDIST_AVAILABLE and not use_window:
        distance = frdist(a, b)
        aggregates = {"frechet_distance": distance}
        # No alignment path available from frchetdist wrapper
        return None, aggregates

    # If use_window: compute frechet on sliding windows and aggregate
    if use_window and window_size is not None:
        min_len = min(a.shape[0], b.shape[0])
        if min_len < window_size:
            return None, {"frechet_distance": float('nan')}

        values = []
        for start in range(0, min_len - window_size + 1, displacement):
            wa = a[start : start + window_size]
            wb = b[start : start + window_size]
            dist, _ = _discrete_frechet(wa, wb)
            values.append(dist)

        aggregates = {"mean": float(np.mean(values)), "median": float(np.median(values)), "std": float(np.std(values))}
        timeseries = np.array(values) if return_timeseries else None

        if timeseries is not None and CONFIG["metrics"].get("save_plots", True) and kwargs.get("out_root"):
            try:
                from src.viz.plots import plot_time_series_for_pair
                os.makedirs(kwargs.get("out_root"), exist_ok=True)
                plot_time_series_for_pair(timeseries, os.path.join(kwargs.get("out_root"), f"frechet_timeseries_{kwargs.get('pair_id','')}.png"), title=f"Fréchet distances ({kwargs.get('pair_id','')})", ylabel="Fréchet Distance")
            except Exception:
                pass

        return timeseries, aggregates

    # Full-trajectory discrete frechet with path
    dist, path = _discrete_frechet(a, b)
    aggregates = {"frechet_distance": dist}
    timeseries = None
    if return_timeseries and path is not None:
        timeseries = np.array([np.linalg.norm(a[i] - b[j]) for i, j in path])
        if CONFIG.get("plots", {}).get("save_timeseries", False) and kwargs.get("out_root"):
            try:
                from src.viz.plots import plot_time_series_for_pair
                os.makedirs(kwargs.get("out_root"), exist_ok=True)
                plot_time_series_for_pair(
                    timeseries,
                    os.path.join(kwargs.get("out_root"), f"frechet_alignment_timeseries_{kwargs.get('pair_id','')}.png"),
                    title=f"Fréchet alignment distances ({kwargs.get('pair_id','')})",
                    ylabel="Fréchet Distance",
                )
            except Exception:
                pass

    return timeseries, aggregates
