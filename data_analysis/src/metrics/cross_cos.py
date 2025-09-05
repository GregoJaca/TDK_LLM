# New metric: cross_cos
import numpy as np
from typing import Tuple, Optional, Dict
from config import CONFIG
from src.viz import plots
import os

class CrossCosMetric:
    """Compute cross-cosine similarity / distance between two trajectories.

    Methods
    - compute_cross_similarity: returns (T_a, T_b) similarity matrix S where S[i,j]=cos(a[i], b[j])
    - cross_distance = 1 - S
    - column_sums (optionally sliding window) -> 1D array of length T_b (summing over rows within window)

    Outputs: saves two plots via src.viz.plots:
    - the cross-distance matrix image saved with pair ids in filename
    - the column-sum timeseries image saved with pair ids in filename
    """

    def __init__(self, use_window: Optional[bool] = None, window_size: Optional[int] = None, window_stride: Optional[int] = None):
        cfg = CONFIG.get("cross_cos", {})
        # params dataclass may exist; order of precedence:
        # explicit args -> top-level cfg keys -> params dataclass -> hardcoded defaults
        params = cfg.get("params")

        # use_window
        if use_window is not None:
            self.use_window = bool(use_window)
        elif "use_window" in cfg:
            self.use_window = bool(cfg.get("use_window"))
        elif params is not None:
            self.use_window = getattr(params, "use_window", False)
        else:
            self.use_window = False

        # window_size
        if window_size is not None:
            self.window_size = int(window_size)
        elif "window_size" in cfg:
            self.window_size = int(cfg.get("window_size"))
        elif params is not None:
            self.window_size = int(getattr(params, "window_size", 5))
        else:
            self.window_size = 5

        # window_stride
        if window_stride is not None:
            self.window_stride = int(window_stride)
        elif "window_stride" in cfg:
            self.window_stride = int(cfg.get("window_stride"))
        elif params is not None:
            self.window_stride = int(getattr(params, "window_stride", 1))
        else:
            self.window_stride = 1

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        denom = np.linalg.norm(x, axis=-1, keepdims=True)
        denom[denom == 0] = 1.0
        return x / denom

    def compute_cross_similarity(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        # a: (Ta, D), b: (Tb, D)
        a_n = self._normalize(a)
        b_n = self._normalize(b)
        # S: (Ta, Tb)
        S = np.matmul(a_n, b_n.T)
        return S

    def compute_column_sums(self, cross_dist: np.ndarray) -> np.ndarray:
        # cross_dist shape: (Ta, Tb); sum over rows for each column optionally with sliding window
        Ta, Tb = cross_dist.shape
        if not self.use_window:
            # simple column sum, with optional stride (downsampling)
            sums = np.nansum(cross_dist, axis=0)
            if self.window_stride and int(self.window_stride) > 1:
                return sums[:: int(self.window_stride)]
            return sums

        # With sliding window: for each column index j, consider rows i in a window around j
        # Define window on rows relative to column index. We'll center window at row index ~= column index.
        w = max(1, int(self.window_size))
        s = max(1, int(self.window_stride))
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

    def compare_trajectories(self, a: np.ndarray, b: np.ndarray, *, return_timeseries: bool = True, pair_id: Optional[str] = None, out_root: Optional[str] = None, **kwargs) -> Tuple[Optional[np.ndarray], Dict[str, float]]:
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

        # compute similarity and distance
        S = self.compute_cross_similarity(a, b)
        cross_dist = 1.0 - S

        # Save matrix image
        os.makedirs(out_root, exist_ok=True)
        matrix_fname = os.path.join(out_root, f"cross_cos_dist_matrix_{pair_id}.png")
        try:
            plots.plot_pairwise_distance_distribution({"pairs": {pair_id: {}}}, matrix_fname, metric_name="cross_cos", aggregate_type="mean")
        except Exception:
            # fallback: simple imshow
            import matplotlib.pyplot as plt
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
        col_sums = self.compute_column_sums(cross_dist)

        timeseries_fname = os.path.join(out_root, f"cross_cos_col_sums_{pair_id}.png")
        try:
            plots.plot_time_series_for_pair(col_sums, timeseries_fname)
        except Exception:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(col_sums)
            plt.title(f'Cross Cos Column Sums ({pair_id})')
            plt.xlabel('Index b')
            plt.ylabel('Sum Distance')
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


def compare_trajectories(a: np.ndarray, b: np.ndarray, *, return_timeseries: bool = True, **kwargs):
    """Module-level wrapper to match the other metrics' interface.

    Accepts (a, b) and returns (timeseries, aggregates). Additional kwargs like
    pair_id or out_root are forwarded to the class method.
    """
    metric = CrossCosMetric()
    return metric.compare_trajectories(a, b, return_timeseries=return_timeseries, **kwargs)
