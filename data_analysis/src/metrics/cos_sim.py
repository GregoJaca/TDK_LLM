import numpy as np
import os
from typing import Tuple, Optional, Dict
from config import CONFIG
import torch

_EPS = 1e-12

def _as_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    return np.asarray(x)

def _normalize_rows(X: np.ndarray):
    norms = np.linalg.norm(X, axis=-1, keepdims=True)
    norms[norms == 0] = _EPS
    return X / norms, norms.squeeze(-1)

def _gaussian_weights(length: int):
    if length <= 1:
        return np.ones(1)
    center = (length - 1) / 2.0
    sigma = length / 3.0
    idx = np.arange(length)
    w = np.exp(-0.5 * ((idx - center) / sigma) ** 2)
    w = w / (w.sum() + 1e-16)
    return w

def compare_trajectories(
    a, b, *, return_timeseries: bool = True, pair_id: Optional[str] = None, out_root: Optional[str] = None, **kwargs
) -> Tuple[Optional[np.ndarray], Dict[str, float]]:
    """Cosine-similarity based sliding-window metric.

    Modes (config via CONFIG["metrics"]["cos_sim"]):
    - centric_mode: "a" | "b" | "both" (default "a")
    - window_agg: "mean" | "min"  (aggregation across vectors in a window; default "mean")
    - gaussian_weight: bool (default False) apply gaussian temporal weighting inside window
    - centric_agg: when centric_mode == 'both', aggregate A- and B-centric distances with 'mean' or 'min' (default 'mean')

    Returns (timeseries_or_None, aggregates_dict)
    """
    a = _as_numpy(a)
    b = _as_numpy(b)

    cfg = CONFIG.get("metrics", {}).get("cos_sim", {})
    sw = CONFIG.get("sliding_window", {})
    use_window = sw.get("use_window", True)
    window_size = sw.get("window_size", 1)
    displacement = sw.get("displacement", 1)

    window_agg = cfg.get("window_agg", "mean")
    gaussian = bool(cfg.get("gaussian_weight", False))
    centric_mode = cfg.get("centric_mode", "a").lower()
    centric_agg = cfg.get("centric_agg", "mean").lower()

    # ensure 2D arrays
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("Input trajectories must be 2D arrays with shape (T, D)")

    min_len = min(a.shape[0], b.shape[0])

    # Fallback: if not using window, behave like per-index cosine (distance = 1 - cos)
    if not use_window or window_size <= 1:
        T = min_len
        A = a[:T]
        B = b[:T]
        A_n, _ = _normalize_rows(A)
        B_n, _ = _normalize_rows(B)
        sims = np.sum(A_n * B_n, axis=-1)
        dists = 1.0 - sims
        agg = {"mean": float(np.nanmean(dists)), "median": float(np.nanmedian(dists)), "std": float(np.nanstd(dists))}

        timeseries = dists if return_timeseries else None
        # optional plotting
        out_root_local = out_root or kwargs.get("out_root")
        if timeseries is not None and out_root_local and CONFIG.get("metrics", {}).get("save_plots", True) and CONFIG.get("plots", {}).get("save_timeseries", False):
            try:
                from src.viz.plots import plot_time_series_for_pair
                os.makedirs(out_root_local, exist_ok=True)
                if pair_id:
                    fname = os.path.join(out_root_local, f"cos_sim_timeseries_{pair_id}.png")
                else:
                    fname = os.path.join(out_root_local, "cos_sim_timeseries.png")
                plot_time_series_for_pair(timeseries, fname, title=f"CosSim distances ({pair_id})" if pair_id else "CosSim distances", ylabel="CosSim Distance")
            except Exception:
                pass
        return timeseries, agg

    if min_len < window_size:
        return None, {"cos_sim": float('nan')}

    # Precompute gaussian weights if requested
    weights = _gaussian_weights(window_size) if gaussian else None

    def compute_a_centric():
        values = []
        positions = []
        for start in range(0, min_len - window_size + 1, displacement):
            # compare a[start] to window in b
            av = a[start]
            wb = b[start : start + window_size]
            av_n, _ = _normalize_rows(av[np.newaxis, :])
            wb_n, _ = _normalize_rows(wb)
            sims = (wb_n @ av_n.squeeze())
            if sims.size == 0:
                continue
            if window_agg == "min":
                best_sim = float(np.nanmax(sims))
                dist = 1.0 - best_sim
            else:
                if weights is not None:
                    mean_sim = float(np.nansum(weights * sims))
                else:
                    mean_sim = float(np.nanmean(sims))
                dist = 1.0 - mean_sim
            values.append(dist)
            positions.append(start)
        return np.array(positions), np.array(values)

    def compute_b_centric():
        values = []
        positions = []
        for start in range(0, min_len - window_size + 1, displacement):
            bv = b[start]
            wa = a[start : start + window_size]
            bv_n, _ = _normalize_rows(bv[np.newaxis, :])
            wa_n, _ = _normalize_rows(wa)
            sims = (wa_n @ bv_n.squeeze())
            if sims.size == 0:
                continue
            if window_agg == "min":
                best_sim = float(np.nanmax(sims))
                dist = 1.0 - best_sim
            else:
                if weights is not None:
                    mean_sim = float(np.nansum(weights * sims))
                else:
                    mean_sim = float(np.nanmean(sims))
                dist = 1.0 - mean_sim
            values.append(dist)
            positions.append(start)
        return np.array(positions), np.array(values)

    timeseries = None
    if centric_mode == "a":
        positions, values = compute_a_centric()
        timeseries = values if return_timeseries else None
    elif centric_mode == "b":
        positions, values = compute_b_centric()
        timeseries = values if return_timeseries else None
    else:  # both
        pos_a, vals_a = compute_a_centric()
        pos_b, vals_b = compute_b_centric()
        # align by start positions (should be identical range)
        if not np.array_equal(pos_a, pos_b):
            # fallback: align on min length
            L = min(len(vals_a), len(vals_b))
            vals_a = vals_a[:L]
            vals_b = vals_b[:L]
        if centric_agg == "min":
            values = np.minimum(vals_a, vals_b)
        else:
            values = 0.5 * (vals_a + vals_b)
        positions = pos_a[: len(values)]
        timeseries = values if return_timeseries else None

    agg = {"mean": float(np.nanmean(timeseries)) if timeseries is not None and timeseries.size > 0 else float('nan'),
           "median": float(np.nanmedian(timeseries)) if timeseries is not None and timeseries.size > 0 else float('nan'),
           "std": float(np.nanstd(timeseries)) if timeseries is not None and timeseries.size > 0 else float('nan')}

    # Save plot if requested
    out_root_local = out_root or kwargs.get("out_root")
    if timeseries is not None and out_root_local and CONFIG.get("metrics", {}).get("save_plots", True) and CONFIG.get("plots", {}).get("save_timeseries", False):
        try:
            from src.viz.plots import plot_time_series_for_pair
            os.makedirs(out_root_local, exist_ok=True)
            if pair_id:
                fname = os.path.join(out_root_local, f"cos_sim_timeseries_{pair_id}.png")
            else:
                fname = os.path.join(out_root_local, "cos_sim_timeseries.png")
            plot_time_series_for_pair(timeseries, fname, title=f"CosSim distances ({pair_id})" if pair_id else "CosSim distances", ylabel="CosSim Distance", x=positions)
        except Exception:
            pass

    return timeseries, agg
