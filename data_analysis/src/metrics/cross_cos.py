# New metric: cross_cos
import numpy as np
from typing import Tuple, Optional, Dict
from config import CONFIG
from src.viz import plots
import os

def _normalize(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=-1, keepdims=True)
    denom[denom == 0] = 1.0
    return x / denom

def _compute_cross_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # a: (Ta, D), b: (Tb, D)
    a_n = _normalize(a)
    b_n = _normalize(b)
    # S: (Ta, Tb)
    S = np.matmul(a_n, b_n.T)
    return S

def _compute_column_sums(cross_dist: np.ndarray, use_window: bool, window_size: int, window_stride: int) -> np.ndarray:
    # cross_dist shape: (Ta, Tb); sum over rows for each column optionally with sliding window
    Ta, Tb = cross_dist.shape
    if not use_window:
        # simple column sum, with optional stride (downsampling)
        sums = np.nansum(cross_dist, axis=0)
        if window_stride and int(window_stride) > 1:
            return sums[:: int(window_stride)]
        return sums

    # With sliding window: for each column index j, consider rows i in a window around j
    # Define window on rows relative to column index. We'll center window at row index ~= column index.
    w = max(1, int(window_size))
    s = max(1, int(window_stride))
    col_sums = []
    for j in range(0, Tb, s):
        # center around j, compute row start/end
        half = w // 2
        start = max(0, j - half)
        end = min(Ta, start + w)
        # adjust start if at end
        start = max(0, end - w)
        col_sums.append(np.nansum(cross_dist[start:end, j]))
    return np.array(col_sums)

def compare_trajectories(a: np.ndarray, b: np.ndarray, *, return_timeseries: bool = True, pair_id: Optional[str] = None, out_root: Optional[str] = None, **kwargs) -> Tuple[Optional[np.ndarray], Dict[str, float]]:
    """Main entry to compute cross-cos distance matrix, save plots, and return timeseries and aggregates.

    Parameters
    - a, b: arrays of shape (T, D)
    - pair_id: optional string used in filenames (e.g., 'ref0_vs_1')
    - out_root: where to save plots; defaults to CONFIG['results_root']
    - return_timeseries: whether to return the 1D column-sum timeseries
    """
    if pair_id is None:
        pair_id = "pair"
    if out_root is None:
        out_root = CONFIG.get("results_root", "results")

    sw_config = CONFIG.get("sliding_window", {})
    use_window = sw_config.get("use_window", False)
    window_size = sw_config.get("window_size", 5)
    window_stride = sw_config.get("displacement", 1)

    # compute similarity and distance
    S = _compute_cross_similarity(a, b)
    cross_dist = 1.0 - S

    # Save matrix image
    if CONFIG["metrics"]["save_plots"]:
        os.makedirs(out_root, exist_ok=True)
        matrix_fname = os.path.join(out_root, f"cross_cos_dist_matrix_{pair_id}.png")
        try:
            plots.plot_pairwise_distance_distribution({"pairs": {pair_id: {}}}, matrix_fname, metric_name="cross_cos", aggregate_type="mean")
        except Exception:
            # fallback: simple imshow
            import matplotlib.pyplot as plt
            print("plot_pairwise_distance_distribution failed for cross_cos; falling back to simple imshow.")
            plt.figure()
            plt.imshow(cross_dist, aspect='auto', origin='lower', cmap='viridis')
            plt.colorbar(label='Cross Cosine Distance')
            plt.title(f'Cross Cos Distance Matrix ({pair_id})')
            plt.xlabel('Index b')
            plt.ylabel('Index a')
            plt.tight_layout()
            plt.savefig(matrix_fname)
            plt.close()

    # compute column sums (optionally windowed)
    col_sums = _compute_column_sums(cross_dist, use_window, window_size, window_stride)

    # Optionally save timeseries if configured globally
    if CONFIG.get("plots", {}).get("save_timeseries", False):
        timeseries_fname = os.path.join(out_root, f"cross_cos_col_sums_{pair_id}.png")
        try:
            plots.plot_time_series_for_pair(col_sums, timeseries_fname, title=f"Cross-Cos Column Sums ({pair_id})", ylabel="Cross Cosine Distance")
        except Exception:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(col_sums)
            plt.title(f'Cross Cos Column Sums ({pair_id})')
            plt.xlabel('Index b')
            plt.ylabel('Cross Cosine Distance (sum)')
            plt.tight_layout()
            plt.savefig(timeseries_fname)
            plt.close()

    aggregates = {
        "mean": float(np.nanmean(col_sums)),
        "median": float(np.nanmedian(col_sums)),
        "std": float(np.nanstd(col_sums))
    }

    timeseries = col_sums if return_timeseries else None
    return timeseries, aggregates