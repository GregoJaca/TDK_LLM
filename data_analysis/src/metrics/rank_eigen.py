







# from dataclasses_json import config
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os
import traceback
import torch
from collections import defaultdict


from typing import Tuple, Dict, Optional
import numpy as np
import torch
import os

from config import CONFIG
from src.viz import plots as viz_plots


def _compute_pca_eigenvectors_torch(X: torch.Tensor, k: Optional[int] = None) -> torch.Tensor:
    X_centered = X - X.mean(dim=0, keepdim=True)
    # SVD on centered data to get principal directions (Vh)
    U, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)
    if k is not None:
        Vh = Vh[:k]
    return Vh


def _cosine_sim_matrix(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    A_norm = A / (A.norm(dim=1, keepdim=True) + 1e-12)
    B_norm = B / (B.norm(dim=1, keepdim=True) + 1e-12)
    return A_norm @ B_norm.T


def sliding_window_rank_deviation(
    a: np.ndarray,
    b: np.ndarray,
    window_size: Optional[int] = None,
    displacement: Optional[int] = None,
    deviation_metric: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    cfg = CONFIG["metrics"]["rank_eigen"]
    sw = CONFIG.get("sliding_window", {})
    window_size = window_size or sw.get("window_size") or cfg.get("sliding_window_size")
    displacement = displacement or sw.get("displacement") or cfg.get("sliding_window_displacement")
    deviation_metric = (deviation_metric or cfg.get("deviation_metric", "rms")).lower()

    # Accept either numpy arrays or torch tensors
    if isinstance(a, torch.Tensor):
        t1 = a.float()
    else:
        t1 = torch.from_numpy(np.asarray(a)).float()
    if isinstance(b, torch.Tensor):
        t2 = b.float()
    else:
        t2 = torch.from_numpy(np.asarray(b)).float()

    min_len = min(t1.shape[0], t2.shape[0])
    if min_len < window_size:
        return np.array([]), np.array([])

    deviations = []
    positions = []
    for start in range(0, min_len - window_size + 1, displacement):
        w1 = t1[start : start + window_size]
        w2 = t2[start : start + window_size]

        k = min(window_size, w1.shape[1])
        v1 = _compute_pca_eigenvectors_torch(w1, k=k)
        v2 = _compute_pca_eigenvectors_torch(w2, k=k)

        sim = _cosine_sim_matrix(v1, v2)  # [k, k]
        # find the closest matching eigenvector in v2 for each eigenvector in v1
        # indices: shape [k], values in [0..k-1]
        indices = torch.argmax(sim, dim=1)
        closest_ranks = (indices + 1).tolist()

        if deviation_metric == "sum_cos_dist":
            # cosine similarity for the matched pairs, then sum (1 - cos_sim)
            row_idx = torch.arange(sim.shape[0])
            cos_sims = sim[row_idx, indices]
            # sum of cosine distances (1 - cos_sim)
            dev = float(torch.sum(1.0 - cos_sims).item())
        else:
            # default: rank-based deviations (rms or mean abs)
            target = np.arange(1, len(closest_ranks) + 1)
            arr = np.array(closest_ranks) - target
            if deviation_metric == "rms":
                dev = float(np.sqrt(np.mean(arr ** 2)))
            else:
                dev = float(np.mean(np.abs(arr)))
        deviations.append(dev)
        positions.append(start + window_size // 2)

    return np.array(positions), np.array(deviations)


def compare_trajectories(
    a: np.ndarray,
    b: np.ndarray,
    *,
    return_timeseries: bool = True,
    pair_id: Optional[str] = None,
    out_root: Optional[str] = None,
    **kwargs,
) -> Tuple[Optional[np.ndarray], Dict[str, float]]:
    """Compare two trajectories by PCA-eigenvector rank alignment.

    Returns (timeseries, aggregates) where timeseries is the sliding-window deviation
    and aggregates contains summary statistics (mean, median, std) of the deviation.
    This mirrors the interface of other metrics in src/metrics.
    """
    cfg = CONFIG["metrics"]["rank_eigen"]

    # Full-trajectory comparison: compute eigenvector rank mapping
    # Accept either numpy arrays or torch tensors
    if isinstance(a, torch.Tensor):
        t1 = a.float()
    else:
        t1 = torch.from_numpy(np.asarray(a)).float()
    if isinstance(b, torch.Tensor):
        t2 = b.float()
    else:
        t2 = torch.from_numpy(np.asarray(b)).float()
    min_len = min(t1.shape[0], t2.shape[0])
    t1_full, t2_full = t1[:min_len], t2[:min_len]

    v1 = _compute_pca_eigenvectors_torch(t1_full)
    v2 = _compute_pca_eigenvectors_torch(t2_full)
    sim_matrix = _cosine_sim_matrix(v1, v2)
    # find closest matches for each eigenvector in v1 -> index in v2
    indices = torch.argmax(sim_matrix, dim=1)
    closest_ranks = (indices + 1).tolist()

    # If requested, compute full-trajectory sum_cos_dist as an aggregate
    deviation_metric_cfg = cfg.get("deviation_metric", "rms").lower()
    full_metric_value = None
    if deviation_metric_cfg == "sum_cos_dist":
        row_idx = torch.arange(sim_matrix.shape[0])
        cos_sims = sim_matrix[row_idx, indices]
        full_metric_value = float(torch.sum(1.0 - cos_sims).item())

    # sliding-window timeseries
    positions, deviations = sliding_window_rank_deviation(
        a, b, window_size=kwargs.get("window_size"), displacement=kwargs.get("displacement")
    )

    # Aggregates
    agg = {}
    if deviations.size > 0:
        agg["mean"] = float(np.mean(deviations))
        agg["median"] = float(np.median(deviations))
        agg["std"] = float(np.std(deviations))
    else:
        agg["mean"] = float(np.nan)
        agg["median"] = float(np.nan)
        agg["std"] = float(np.nan)
    # include full-trajectory sum_cos_dist when requested
    if full_metric_value is not None:
        agg["sum_cos_dist"] = full_metric_value

    # Save plots if requested and out_root given
    if out_root and CONFIG["metrics"].get("save_plots", True):
        plots_dir = out_root
        os.makedirs(plots_dir, exist_ok=True)

        # full-rank scatter
        try:
            if pair_id:
                full_plot_path = os.path.join(plots_dir, f"rank_eigen_pca_{pair_id}.png")
            else:
                full_plot_path = os.path.join(plots_dir, "rank_eigen_pca.png")
            viz_plots.plot_rank_eigen_full(closest_ranks, full_plot_path, traj_indices=pair_id)
        except Exception:
            # avoid breaking metric computation if plotting fails
            pass

        # sliding plot
        if positions.size > 0 and CONFIG.get("plots", {}).get("save_timeseries", False):
            try:
                if pair_id:
                    sliding_plot_path = os.path.join(plots_dir, f"rank_eigen_pca_{pair_id}_sliding.png")
                else:
                    sliding_plot_path = os.path.join(plots_dir, "rank_eigen_pca_sliding.png")
                viz_plots.plot_rank_eigen_sliding(positions, deviations, sliding_plot_path, metric_cfg=cfg)
            except Exception:
                pass

    timeseries = deviations if return_timeseries and deviations.size > 0 else None
    return timeseries, agg





