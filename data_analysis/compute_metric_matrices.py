"""Compute symmetric per-trajectory metric matrices for each metric and embedding.

This script reproduces the input selection behavior from `run_all_experiments.py` by
iterating `EMBEDDING_METHODS` and Experiment parameters. For each embedding it loads
the tensor (n, T, D) and, for each trajectory, computes a square symmetric matrix
of pairwise metric values between time windows (or single timesteps) of that
trajectory. Matrices are saved as .pt and plotted as PNGs.

Uses existing `src.io.loader.load_tensor` and metric modules in `src.metrics`.

Usage: run directly. Flags at top of file control whether to save matrices and plots.
"""
from __future__ import annotations

import importlib
import os
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch

from config import CONFIG
from src.io.loader import load_tensor
# Read embedding config here rather than importing run_all_experiments to avoid circular imports
EMBEDDING_CONFIG = CONFIG.get('EMBEDDING_CONFIG', {})
EMBEDDING_METHODS = EMBEDDING_CONFIG.get('embedding_methods', [])
INPUT_MODE = EMBEDDING_CONFIG.get('input_mode', 'single_file')
INPUT_TEMPLATE = EMBEDDING_CONFIG.get('input_template', '{embed}.pt')
from src.utils.logging import get_logger

logger = get_logger(__name__)


# Options (simple knobs)
SAVE_MATRICES = False
# Default plot saving follows the user config flag to avoid unexpected logs
SAVE_PLOTS = bool(CONFIG.get('plots', {}).get('save_histograms', True))
TRAJECTORIES_TO_PROCESS: Optional[list[int]] = [0,1,2,3]

def _make_results_dirs(base: str) -> str:
    """Return a single metric_matrices folder under base for all outputs.

    This flattens plotting output into one folder as requested.
    """
    # Preserve backward-compatible single-folder behavior when called with a
    # top-level results root. If `base` already ends with a metric name or
    # otherwise points to the intended output directory, don't append extra
    # folders. The caller typically passes `results_root` and we want files to
    # go into `results_root/<metric_name>` (handled by callers). For any other
    # case, create the supplied base directory.
    out_dir = base
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _select_scalar_from_aggregates(agg: Dict[str, Any]) -> float:
    """Pick a sensible scalar from a metric aggregates dict.

    Preference order: 'mean' -> first numeric value in dict -> nan
    """
    if not agg:
        return float("nan")
    if "mean" in agg:
        try:
            return float(agg["mean"])
        except Exception:
            pass
    # common named scalars
    for key in ("dtw_distance", "hausdorff_distance", "frechet_distance", "sum_cos_dist"):
        if key in agg:
            try:
                return float(agg[key])
            except Exception:
                pass
    # fall back to first numeric value
    for v in agg.values():
        try:
            return float(v)
        except Exception:
            continue
    return float("nan")


def _compute_matrix_for_trajectory(
    traj: np.ndarray,
    metric_mod: Any,
    use_window: bool,
    window_size: int,
    displacement: int,
    out_root: str,
    metric_name: str,
    traj_idx: int,
) -> np.ndarray:
    T = traj.shape[0]

    if use_window and window_size is not None and window_size > 0 and T >= window_size:
        starts = list(range(0, T - window_size + 1, displacement))
        get_segment = lambda s: traj[s : s + window_size]
    else:
        # per-timestep windows
        starts = list(range(0, T))
        get_segment = lambda s: traj[s : s + 1]

    L = len(starts)
    M = np.full((L, L), np.nan, dtype=float)

    # compute only upper triangle and diagonal, mirror to ensure symmetry
    for i_idx, si in enumerate(starts):
        seg_i = get_segment(si)
        for j_idx in range(i_idx, L):
            sj = starts[j_idx]
            seg_j = get_segment(sj)

            # Call metric: prefer not to produce timeseries in this call
            # Some metric implementations read CONFIG['sliding_window'] directly.
            prev_sw = CONFIG.get('sliding_window', None)
            try:
                CONFIG['sliding_window'] = dict(use_window=use_window, window_size=window_size, displacement=displacement)
                try:
                    _, agg_ij = metric_mod.compare_trajectories(
                        seg_i,
                        seg_j,
                        return_timeseries=False,
                        out_root=None,
                        pair_id=None,
                        sliding_window=CONFIG.get('sliding_window', {}),
                        window_size=window_size,
                        displacement=displacement,
                        use_window=use_window,
                    )
                except Exception as e:
                    logger.error(f"Error computing metric {metric_name} for traj {traj_idx} (i->j): {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    agg_ij = {}
                try:
                    _, agg_ji = metric_mod.compare_trajectories(
                        seg_j,
                        seg_i,
                        return_timeseries=False,
                        out_root=None,
                        pair_id=None,
                        sliding_window=CONFIG.get('sliding_window', {}),
                        window_size=window_size,
                        displacement=displacement,
                        use_window=use_window,
                    )
                except Exception as e:
                    logger.error(f"Error computing metric {metric_name} for traj {traj_idx} (j->i): {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    agg_ji = {}
            finally:
                if prev_sw is None:
                    CONFIG.pop('sliding_window', None)
                else:
                    CONFIG['sliding_window'] = prev_sw

            v1 = _select_scalar_from_aggregates(agg_ij)
            v2 = _select_scalar_from_aggregates(agg_ji)

            # If aggregates are NaN, try to compute timeseries and derive a scalar (fallback)
            if np.isnan(v1):
                try:
                    ts, _ = metric_mod.compare_trajectories(seg_i, seg_j, return_timeseries=True)
                    if ts is not None:
                        v1 = float(np.nanmean(np.asarray(ts)))
                except Exception:
                    pass
            if np.isnan(v2):
                try:
                    ts2, _ = metric_mod.compare_trajectories(seg_j, seg_i, return_timeseries=True, out_root=None, pair_id=None)
                    if ts2 is not None:
                        v2 = float(np.nanmean(np.asarray(ts2)))
                except Exception:
                    pass

            # average, handling nans
            vals = [x for x in (v1, v2) if not (x is None or np.isnan(x))]
            if vals:
                val = float(np.mean(vals))
            else:
                val = float("nan")

            M[i_idx, j_idx] = val
            M[j_idx, i_idx] = val

    return M


def _plot_matrix(M: np.ndarray, out_path: str, title: Optional[str] = None, sweep_param_value=None):
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 6))
        plt.imshow(M, aspect="equal", origin="lower", cmap="viridis")
        plt.colorbar()
        if title:
            if sweep_param_value is not None:
                title += f" - window_size={sweep_param_value}"
            plt.title(title)
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
    except Exception:
        # plotting should not be fatal
        pass


def _compute_cross_cos_matrix(
    traj: np.ndarray,
    window_size: int,
    displacement: int,
    use_window: bool,
) -> np.ndarray:
    """Compute cross-cos distance matrix for a trajectory, optionally windowed.

    If `use_window` is True and `window_size` fits, each window is represented by
    the mean vector across the window; otherwise each timestep is used.
    """
    X = np.asarray(traj, dtype=float)
    T = X.shape[0]

    if use_window and window_size is not None and window_size > 0 and T >= window_size:
        starts = list(range(0, T - window_size + 1, displacement))
        reps = [np.mean(X[s : s + window_size], axis=0) for s in starts]
    else:
        starts = list(range(0, T))
        reps = [X[s] for s in starts]

    if len(reps) == 0:
        return np.empty((0, 0))

    A = np.vstack([np.asarray(r, dtype=float).reshape(1, -1) for r in reps])
    norms = np.linalg.norm(A, axis=-1, keepdims=True)
    norms[norms == 0] = 1e-12
    An = A / norms
    S = An @ An.T
    dist = 1.0 - S
    dist = 0.5 * (dist + dist.T)
    return dist


def _compute_rank_eigen_matrix(
    traj: np.ndarray,
    window_size: int,
    displacement: int,
    deviation_metric: str = "rms",
    device: Optional[str] = None,
) -> np.ndarray:
    """Compute pairwise rank-eigen deviation matrix for a trajectory.

    Behavior mirrors src.metrics.rank_eigen.compare_trajectories but computes a
    full L x L matrix by caching PCA eigenvectors per window and comparing them.
    """
    X = np.asarray(traj, dtype=float)
    T = X.shape[0]

    # Determine windows
    if window_size is not None and window_size > 0 and T >= window_size and CONFIG.get("sliding_window", {}).get("use_window", False):
        starts = list(range(0, T - window_size + 1, displacement))
        get_segment = lambda s: X[s : s + window_size]
    else:
        starts = list(range(0, T))
        get_segment = lambda s: X[s : s + 1]

    L = len(starts)
    if L == 0:
        return np.empty((0, 0))

    # Prepare device
    use_gpu = False
    if device is None:
        use_gpu = bool(CONFIG.get("gpu", {}).get("use_gpu_for_pca", False)) and torch.cuda.is_available()
    else:
        use_gpu = device.lower().startswith("cuda") if isinstance(device, str) else False
    dev = torch.device("cuda" if use_gpu else "cpu")

    # Precompute eigenvectors per window (as torch tensors on device)
    eig_list: List[torch.Tensor] = []
    for s in starts:
        seg = get_segment(s)
        # seg shape (W, D)
        t = torch.from_numpy(np.asarray(seg)).float().to(dev)
        # center
        t_centered = t - t.mean(dim=0, keepdim=True)
        try:
            U, S, Vh = torch.linalg.svd(t_centered, full_matrices=False)
        except RuntimeError:
            # fallback to CPU SVD if GPU fails
            t_cpu = t_centered.cpu()
            U, S, Vh = torch.linalg.svd(t_cpu, full_matrices=False)
            Vh = Vh.to(dev)

        k = min(t.shape[0], t.shape[1])
        # keep top-k eigenvectors (rows of Vh)
        V = Vh[:k]  # shape (k, D)
        eig_list.append(V)

    # Compute LxL matrix
    M = np.full((L, L), np.nan, dtype=float)

    for i in range(L):
        v1 = eig_list[i]
        # normalize rows
        v1n = v1 / (v1.norm(dim=1, keepdim=True) + 1e-12)
        for j in range(i, L):
            v2 = eig_list[j]
            v2n = v2 / (v2.norm(dim=1, keepdim=True) + 1e-12)

            # similarity matrix (k1 x k2)
            sim = v1n @ v2n.t()
            # for each eigenvector in v1, find best match in v2
            row_idx = torch.arange(sim.shape[0], device=sim.device)
            indices = torch.argmax(sim, dim=1)
            cos_sims = sim[row_idx, indices]

            if deviation_metric == "sum_cos_dist":
                dev_val = float(torch.sum(1.0 - cos_sims).cpu().numpy())
            else:
                # rank deviation: indices->1-based ranks
                closest_ranks = (indices.cpu().numpy() + 1).astype(float)
                target = np.arange(1, len(closest_ranks) + 1, dtype=float)
                arr = closest_ranks - target
                if deviation_metric == "rms":
                    dev_val = float(np.sqrt(np.mean(arr ** 2)))
                else:
                    dev_val = float(np.mean(np.abs(arr)))

            M[i, j] = dev_val
            M[j, i] = dev_val

    return M


def main(input_path, results_root, save_matrices: bool = SAVE_MATRICES, save_plots: bool = SAVE_PLOTS, sweep_param_value=None):
    logger.info(f"compute_metric_matrices.main called with input_path={input_path}, results_root={results_root}, sweep_param_value={sweep_param_value}")
    sw = CONFIG.get("sliding_window", {})
    use_window = bool(sw.get("use_window", False))
    window_size = int(sw.get("window_size", 1))
    displacement = int(sw.get("displacement", 1))

    logger.info(f"Processing input={input_path}")

    # load tensor(s)
    trajectories = []
    if INPUT_MODE == 'per_trajectory':
        # input_path may be a run folder
        base_folder = input_path if os.path.isdir(input_path) else os.path.dirname(input_path)
        current_embed = CONFIG.get('current_embed_raw')
        pairwise_cfg = CONFIG.get('pairwise', {})
        # decide which indices to load
        indices_to_load = None
        if not pairwise_cfg.get('compute_all_pairs', False):
            ref_idx = pairwise_cfg.get('reference_index', None)
            if ref_idx is None:
                pairs = pairwise_cfg.get('pairs_to_plot', [])
                idxs = set()
                for p in pairs:
                    try:
                        idxs.add(int(p[0])); idxs.add(int(p[1]))
                    except Exception:
                        pass
                if idxs:
                    indices_to_load = sorted(list(idxs))

        # If no specific indices requested, try to load all .pt files in folder
        if indices_to_load is None:
            files = sorted([f for f in os.listdir(base_folder) if f.endswith('.pt')])
            for f in files:
                p = os.path.join(base_folder, f)
                try:
                    t = load_tensor(p)
                    trajectories.append(t)
                except Exception:
                    logger.warning(f"Failed to load {p}; skipping")
        else:
            input_template = INPUT_TEMPLATE
            for i in indices_to_load:
                fname = input_template.format(embed=current_embed, i=i)
                p = os.path.join(base_folder, fname)
                if os.path.exists(p):
                    try:
                        t = load_tensor(p)
                        trajectories.append(t)
                    except Exception:
                        logger.warning(f"Failed to load {p}; skipping")
                else:
                    logger.warning(f"Per-trajectory file {p} not found; skipping.")

        if len(trajectories) == 0:
            logger.error(f"No trajectories loaded from {input_path}; skipping.")
            return

        # convert to numpy array of shape (n, T, D)
        data = np.stack([np.asarray(t) for t in trajectories], axis=0)
        n, T, D = data.shape
    else:
        # single-file behavior
        try:
            tensor = load_tensor(input_path)
            logger.info(f"Loaded tensor from {input_path}")
        except Exception as e:
            logger.error(f"Failed to load {input_path}: {e}")
            return

        if isinstance(tensor, torch.Tensor):
            data = tensor.cpu().numpy()
        else:
            data = np.asarray(tensor)

        if data.ndim != 3:
            logger.error(f"Unexpected tensor shape {data.shape} for {input_path}; expected (n,T,D). Skipping.")
            return

        n, T, D = data.shape

    # Determine metrics to run: build list from per-metric 'enabled' flags only.
    # The project historically allowed an explicit 'available' whitelist; that
    # key has been removed from policy here and we treat per-metric
    # CONFIG['metrics'][<metric>]['enabled'] as the source of truth.
    cfg_metrics = CONFIG.get("metrics", {})
    metrics_list: List[str] = []
    for k, v in cfg_metrics.items():
        # skip non-dict entries
        if not isinstance(v, dict):
            continue
        try:
            if bool(v.get('enabled', False)):
                metrics_list.append(k)
        except Exception:
            continue
    logger.info(f"compute_metric_matrices: running metrics_list (from 'enabled' flags)={metrics_list}")

    for metric_name in metrics_list:
        # Only run metrics explicitly enabled in CONFIG
        metric_cfg = CONFIG.get("metrics", {}).get(metric_name, {})
        if not metric_cfg.get("enabled", True):
            logger.info(f"Skipping metric {metric_name} (enabled=False)")
            continue
        # Try to import metric module; allow some common name fallbacks
        metric_mod = None
        tried_names = []
        candidates = [metric_name]
        # common fallback: 'dtw' -> 'dtw_fast'
        if metric_name == 'dtw':
            candidates.append('dtw_fast')
        if not metric_name.endswith('_fast'):
            candidates.append(metric_name + '_fast')

        for cand in candidates:
            try:
                tried_names.append(cand)
                metric_mod = importlib.import_module(f"src.metrics.{cand}")
                break
            except Exception:
                metric_mod = None

        if metric_mod is None:
            logger.error(f"Failed to import metric {metric_name} (tried: {tried_names}). Skipping.")
            continue

        logger.info(f" Computing metric '{metric_name}' for {n} trajectories (this may take a while)...")

        for traj_idx in range(n):
            if TRAJECTORIES_TO_PROCESS is not None and traj_idx not in TRAJECTORIES_TO_PROCESS:
                continue
            traj = data[traj_idx]
            
            if sweep_param_value is not None:
                out_root = os.path.join(results_root, metric_name)
                os.makedirs(out_root, exist_ok=True)
            else:
                out_root = _make_results_dirs(results_root)

            # Special-case: cross_cos wants a single full T x T matrix per trajectory
            if metric_name == "cross_cos":
                try:
                    M = _compute_cross_cos_matrix(traj, window_size=window_size, displacement=displacement, use_window=use_window)
                    if save_matrices:
                        torch.save(torch.tensor(M), os.path.join(out_root, f"{metric_name}_traj{traj_idx}.pt"))
                    if save_plots and CONFIG.get('plots', {}).get('save_histograms', True):
                        if sweep_param_value is not None:
                            plot_path = os.path.join(out_root, f"{metric_name}_traj{traj_idx}_window_size_{sweep_param_value}.png")
                        else:
                            plot_path = os.path.join(out_root, f"{metric_name}_traj{traj_idx}.png")
                        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
                        logger.info(f"Saving plot to {plot_path}")
                        _plot_matrix(M, plot_path, title=f"{metric_name} matrix traj {traj_idx}", sweep_param_value=sweep_param_value)
                    # skip the generic per-window pairwise routine
                    continue
                except Exception as e:
                    logger.error(f"Failed to compute full cross_cos matrix for traj {traj_idx}: {e}")

            # Special-case: rank_eigen compute full matrix via cached PCA eigenvectors
            if metric_name == "rank_eigen":
                try:
                    rank_cfg = CONFIG.get("metrics", {}).get("rank_eigen", {})
                    deviation_metric = rank_cfg.get("deviation_metric", "rms")
                    M = _compute_rank_eigen_matrix(
                        traj,
                        window_size=window_size,
                        displacement=displacement,
                        deviation_metric=deviation_metric,
                    )
                    if save_matrices:
                        torch.save(torch.tensor(M), os.path.join(out_root, f"{metric_name}_traj{traj_idx}.pt"))
                    if save_plots and CONFIG.get('plots', {}).get('save_histograms', True):
                        if sweep_param_value is not None:
                            plot_path = os.path.join(out_root, f"{metric_name}_traj{traj_idx}_window_size_{sweep_param_value}.png")
                        else:
                            plot_path = os.path.join(out_root, f"{metric_name}_traj{traj_idx}.png")
                        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
                        logger.info(f"Saving plot to {plot_path}")
                        _plot_matrix(M, plot_path, title=f"{metric_name} matrix traj {traj_idx}", sweep_param_value=sweep_param_value)
                    continue
                except Exception as e:
                    logger.error(f"Failed to compute rank_eigen matrix for traj {traj_idx}: {e}")

            M = _compute_matrix_for_trajectory(
                traj=traj,
                metric_mod=metric_mod,
                use_window=use_window,
                window_size=window_size,
                displacement=displacement,
                out_root=out_root,
                metric_name=metric_name,
                traj_idx=traj_idx,
            )

            if save_matrices:
                try:
                    torch.save(torch.tensor(M), os.path.join(out_root, f"{metric_name}_traj{traj_idx}.pt"))
                except Exception as e:
                    logger.error(f"Failed to save matrix for {metric_name} traj {traj_idx}: {e}")

            if save_plots:
                if sweep_param_value is not None:
                    plot_path = os.path.join(out_root, f"{metric_name}_traj{traj_idx}_window_size_{sweep_param_value}.png")
                else:
                    plot_path = os.path.join(out_root, f"{metric_name}_traj{traj_idx}.png")
                logger.info(f"Saving plot to {plot_path}")
                _plot_matrix(M, plot_path, title=f"{metric_name} matrix traj {traj_idx}", sweep_param_value=sweep_param_value)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to the input tensor file")
    parser.add_argument("--results", type=str, default=None, help="Path to the results directory (defaults to the input path if omitted)")
    args = parser.parse_args()
    results_root = args.results if args.results is not None else args.input
    main(input_path=args.input, results_root=results_root)
