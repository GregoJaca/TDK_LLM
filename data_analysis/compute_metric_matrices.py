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

# Reuse embedded list & Experiment definition from run_all_experiments
from run_all_experiments import EMBEDDING_METHODS, Experiment


# Options (simple knobs)
SAVE_MATRICES = False
SAVE_PLOTS = True
TRAJECTORIES_TO_PROCESS: Optional[list[int]] = [0]

def _make_results_dirs(base: str) -> str:
    """Return a single metric_matrices folder under base for all outputs.

    This flattens plotting output into one folder as requested.
    """
    out_dir = os.path.join(base, "metric_matrices")
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
            try:
                # pass out_root=None to avoid metric modules saving their own plots
                _, agg_ij = metric_mod.compare_trajectories(seg_i, seg_j, return_timeseries=False, out_root=None, pair_id=None)
            except Exception:
                agg_ij = {}
            try:
                _, agg_ji = metric_mod.compare_trajectories(seg_j, seg_i, return_timeseries=False, out_root=None, pair_id=None)
            except Exception:
                agg_ji = {}

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


def _plot_matrix(M: np.ndarray, out_path: str, title: Optional[str] = None):
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 6))
        plt.imshow(M, aspect="equal", origin="lower", cmap="viridis")
        plt.colorbar()
        if title:
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


def main(save_matrices: bool = SAVE_MATRICES, save_plots: bool = SAVE_PLOTS):
    sw = CONFIG.get("sliding_window", {})
    use_window = bool(sw.get("use_window", False))
    window_size = int(sw.get("window_size", 1))
    displacement = int(sw.get("displacement", 1))

    repo_root = os.path.abspath(os.path.dirname(__file__))

    repo_root = os.path.abspath(os.path.dirname(__file__))

    for rrr in Experiment.RADII:
        for TEMPERATURE in Experiment.TEMPS:
            for embedder in EMBEDDING_METHODS:
                embed_name = embedder.replace("/", "_")
                input_path = os.path.normpath(
                    f"C:/Users/grego/OneDrive/Documents/BME_UNI_WORK/TDK_2025/git_repo/TDK_LLM/runs_aug/launch_aug/childhood_personality_development_{TEMPERATURE}_{rrr}/{embed_name}.pt"
                )
                results_root = os.path.normpath(
                    f"C:/Users/grego/OneDrive/Documents/BME_UNI_WORK/TDK_2025/git_repo/TDK_LLM/runs_aug/launch_aug/childhood_personality_development_{TEMPERATURE}_{rrr}/results_rp/{embed_name}"
                )

                # ensure results dir
                os.makedirs(results_root, exist_ok=True)

                prev_input = CONFIG.get("input_path")
                prev_results = CONFIG.get("results_root")
                CONFIG["input_path"] = input_path
                CONFIG["results_root"] = results_root

                print(f"Processing embedder={embedder} input={input_path}")

                # load tensor
                try:
                    tensor = load_tensor(input_path)
                except Exception as e:
                    print(f"Failed to load {input_path}: {e}")
                    # restore and continue
                    if prev_input is None:
                        CONFIG.pop("input_path", None)
                    else:
                        CONFIG["input_path"] = prev_input
                    if prev_results is None:
                        CONFIG.pop("results_root", None)
                    else:
                        CONFIG["results_root"] = prev_results
                    continue

                # Expect (n, T, D)
                if isinstance(tensor, torch.Tensor):
                    data = tensor.cpu().numpy()
                else:
                    data = np.asarray(tensor)

                if data.ndim != 3:
                    print(f"Unexpected tensor shape {data.shape} for {input_path}; expected (n,T,D). Skipping.")
                    # restore
                    if prev_input is None:
                        CONFIG.pop("input_path", None)
                    else:
                        CONFIG["input_path"] = prev_input
                    if prev_results is None:
                        CONFIG.pop("results_root", None)
                    else:
                        CONFIG["results_root"] = prev_results
                    continue

                n, T, D = data.shape

                metrics_list: List[str] = CONFIG.get("metrics", {}).get("available", [])

                for metric_name in metrics_list:
                    # Only run metrics explicitly enabled in CONFIG
                    metric_cfg = CONFIG.get("metrics", {}).get(metric_name, {})
                    if not metric_cfg.get("enabled", True):
                        print(f"Skipping metric {metric_name} (enabled=False)")
                        continue
                    try:
                        metric_mod = importlib.import_module(f"src.metrics.{metric_name}")
                    except Exception as e:
                        print(f"Failed to import metric {metric_name}: {e}. Skipping.")
                        continue

                    print(f" Computing metric '{metric_name}' for {n} trajectories (this may take a while)...")

                    for traj_idx in range(n):
                        if TRAJECTORIES_TO_PROCESS is not None and traj_idx not in TRAJECTORIES_TO_PROCESS:
                            continue
                        traj = data[traj_idx]
                        # single shared folder for all metric matrices/plots
                        out_root = _make_results_dirs(results_root)
                        # Special-case: cross_cos wants a single full T x T matrix per trajectory
                        if metric_name == "cross_cos":
                            try:
                                M = _compute_cross_cos_matrix(traj, window_size=window_size, displacement=displacement, use_window=use_window)
                                if save_matrices:
                                    torch.save(torch.tensor(M), os.path.join(out_root, f"{metric_name}_traj{traj_idx}.pt"))
                                if save_plots:
                                    plot_path = os.path.join(out_root, f"{metric_name}_traj{traj_idx}.png")
                                    _plot_matrix(M, plot_path, title=f"{metric_name} matrix traj {traj_idx}")
                                # skip the generic per-window pairwise routine
                                continue
                            except Exception as e:
                                print(f"Failed to compute full cross_cos matrix for traj {traj_idx}: {e}")

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
                                if save_plots:
                                    plot_path = os.path.join(out_root, f"{metric_name}_traj{traj_idx}.png")
                                    _plot_matrix(M, plot_path, title=f"{metric_name} matrix traj {traj_idx}")
                                continue
                            except Exception as e:
                                print(f"Failed to compute rank_eigen matrix for traj {traj_idx}: {e}")

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
                                print(f"Failed to save matrix for {metric_name} traj {traj_idx}: {e}")

                        if save_plots:
                            plot_path = os.path.join(out_root, f"{metric_name}_traj{traj_idx}.png")
                            _plot_matrix(M, plot_path, title=f"{metric_name} matrix traj {traj_idx}")

                # restore CONFIG
                if prev_input is None:
                    CONFIG.pop("input_path", None)
                else:
                    CONFIG["input_path"] = prev_input
                if prev_results is None:
                    CONFIG.pop("results_root", None)
                else:
                    CONFIG["results_root"] = prev_results


if __name__ == "__main__":
    main()
