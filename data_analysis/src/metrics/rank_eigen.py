







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
        closest_ranks = []
        for ii in range(sim.shape[0]):
            _, indices = torch.sort(sim[ii], descending=True)
            closest_idx = indices[0].item()
            closest_ranks.append(closest_idx + 1)

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

    closest_ranks = []
    for ii in range(sim_matrix.shape[0]):
        _, indices = torch.sort(sim_matrix[ii], descending=True)
        closest_idx = indices[0].item()
        closest_ranks.append(closest_idx + 1)

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

    # Save plots if requested and out_root given
    if out_root and CONFIG["metrics"].get("save_plots", True):
        plots_dir = out_root
        os.makedirs(plots_dir, exist_ok=True)

        # full-rank scatter
        try:
            full_plot_path = os.path.join(plots_dir, f"rank_eigen_pca_{pair_id}.png") if pair_id else os.path.join(plots_dir, "rank_eigen_pca.png")
            viz_plots.plot_rank_eigen_full(closest_ranks, full_plot_path, traj_indices=pair_id)
        except Exception:
            # avoid breaking metric computation if plotting fails
            pass

        # sliding plot
        if positions.size > 0 and CONFIG.get("plots", {}).get("save_timeseries", False):
            try:
                sliding_plot_path = os.path.join(plots_dir, f"rank_eigen_pca_{pair_id}_sliding.png") if pair_id else os.path.join(plots_dir, "rank_eigen_pca_sliding.png")
                viz_plots.plot_rank_eigen_sliding(positions, deviations, sliding_plot_path, metric_cfg=cfg)
            except Exception:
                pass

    timeseries = deviations if return_timeseries and deviations.size > 0 else None
    return timeseries, agg





