import os
import numpy as np
import csv
from config import CONFIG
from .config import LyapunovConfig
from .averaging import average_curves, detect_no_rise
from .fitters import fit_timeseries
from .plotting import plot_curves, plot_mean_and_fit, plot_lyapunov_time_series, plot_saturation_detection, plot_distance_time_series, plot_midpoint_distribution
from .results import make_output_dir, save_results, save_npz
import json


def detect_curve_transition_points(curve: np.ndarray, sat_cfg: dict, total_length: int = None):
    arr = np.asarray(curve, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"curve must be 1D, got shape={arr.shape}")

    T = int(arr.shape[0]) if total_length is None else int(total_length)
    if T <= 0:
        return {
            "jump_index": 0,
            "second_jump_index": None,
            "saturation_index": 0,
            "baseline": float("nan"),
            "plateau": float("nan"),
            "midpoint": float("nan"),
            "smooth_window": 1,
            "detect_in_log_scale": bool(sat_cfg.get("detect_in_log_scale", True)),
            "smooth_curve_detect": arr.copy(),
            "smooth_curve_plot": arr.copy(),
            "baseline_plot": float("nan"),
            "plateau_plot": float("nan"),
            "midpoint_plot": float("nan"),
        }

    baseline_frac = float(sat_cfg.get("baseline_frac", 0.05))
    plateau_frac = float(sat_cfg.get("plateau_frac", 0.2))
    midpoint_frac = float(sat_cfg.get("midpoint_frac", 0.5))
    detect_in_log_scale = bool(sat_cfg.get("detect_in_log_scale", True))
    log_eps = float(sat_cfg.get("log_eps", 1e-12))

    baseline_len = max(3, int(T * baseline_frac))
    plateau_len = max(3, int(T * plateau_frac))

    detect_curve = np.log(np.maximum(arr, log_eps)) if detect_in_log_scale else arr
    smooth_w = max(3, int(round(T * 0.03)))
    if smooth_w % 2 == 0:
        smooth_w += 1

    pad = smooth_w // 2
    padded = np.pad(detect_curve, (pad, pad), mode="edge")
    kernel = np.ones(smooth_w, dtype=float) / float(smooth_w)
    smooth = np.convolve(padded, kernel, mode="valid")

    if T <= 3:
        jump_idx = 1 if T > 1 else 0
        second_jump_idx = None
        deriv = np.zeros(max(0, T - 1), dtype=float)
    else:
        deriv = np.abs(np.diff(smooth))
        jump_idx = int(np.argmax(deriv)) + 1

        # Second jump detection: after first jump, require a settled region and
        # then detect a sustained rise toward the final plateau.
        min_sep = max(2, int(round(T * float(sat_cfg.get("second_jump_min_sep_frac", 0.05)))))
        settle_len = max(5, int(round(T * float(sat_cfg.get("second_jump_settle_frac", 0.08)))))
        level_frac = float(sat_cfg.get("second_jump_level_frac", 0.5))
        min_delta_frac = float(sat_cfg.get("second_jump_min_delta_frac", 0.08))
        consecutive = max(1, int(sat_cfg.get("second_jump_consecutive_points", 3)))

        final_plateau = float(np.nanmedian(smooth[max(0, T - plateau_len):]))
        start2 = min(T - 1, jump_idx + min_sep)

        if start2 + settle_len >= T:
            second_jump_idx = None
        else:
            mid_plateau = float(np.nanmedian(smooth[start2:start2 + settle_len]))
            total_range = float(np.nanmax(smooth) - np.nanmin(smooth))
            delta_to_final = float(final_plateau - mid_plateau)
            required_delta = max(1e-12, min_delta_frac * max(total_range, 1e-12))

            if delta_to_final <= required_delta:
                second_jump_idx = None
            else:
                target2 = mid_plateau + level_frac * delta_to_final
                post = smooth[start2:]
                hits = post >= target2

                second_jump_idx = None
                if hits.size > 0:
                    run = 0
                    for k, ok in enumerate(hits):
                        run = run + 1 if ok else 0
                        if run >= consecutive:
                            second_jump_idx = int(start2 + k - consecutive + 1)
                            break

                if second_jump_idx is None:
                    deriv2 = np.abs(np.diff(smooth[start2:]))
                    if deriv2.size > 0 and np.isfinite(np.nanmax(deriv2)):
                        second_jump_idx = int(start2 + np.argmax(deriv2) + 1)

    left_end = max(baseline_len, jump_idx)
    right_start = min(jump_idx, T - plateau_len)
    if right_start < 0:
        right_start = 0

    baseline = float(np.nanmedian(smooth[:left_end])) if left_end > 0 else float(np.nanmedian(smooth[:baseline_len]))
    plateau = float(np.nanmedian(smooth[right_start:])) if right_start < T else float(np.nanmedian(smooth[-plateau_len:]))
    midpoint = baseline + midpoint_frac * (plateau - baseline)

    search_start = max(0, jump_idx - max(2, int(0.05 * T)))
    if plateau >= baseline:
        local = np.where(smooth[search_start:] >= midpoint)[0]
    else:
        local = np.where(smooth[search_start:] <= midpoint)[0]
    if local.size > 0:
        sat_idx = int(search_start + local[0])
    else:
        sat_idx = int(jump_idx)

    if detect_in_log_scale:
        baseline_plot = float(np.exp(np.clip(baseline, -700, 700)))
        plateau_plot = float(np.exp(np.clip(plateau, -700, 700)))
        midpoint_plot = float(np.exp(np.clip(midpoint, -700, 700)))
        smooth_plot = np.exp(np.clip(smooth, -700, 700))
    else:
        baseline_plot = float(baseline)
        plateau_plot = float(plateau)
        midpoint_plot = float(midpoint)
        smooth_plot = smooth

    return {
        "jump_index": int(jump_idx),
        "second_jump_index": int(second_jump_idx) if second_jump_idx is not None else None,
        "saturation_index": int(sat_idx),
        "baseline": float(baseline),
        "plateau": float(plateau),
        "midpoint": float(midpoint),
        "smooth_window": int(smooth_w),
        "detect_in_log_scale": bool(detect_in_log_scale),
        "smooth_curve_detect": smooth,
        "smooth_curve_plot": smooth_plot,
        "baseline_plot": float(baseline_plot),
        "plateau_plot": float(plateau_plot),
        "midpoint_plot": float(midpoint_plot),
    }


def process_file(path, cfg: LyapunovConfig = None, outdir: str = None):
    cfg = cfg or LyapunovConfig.from_global()
    data = np.load(path)
    timeseries_raw = data["timeseries"]
    pair_indices = data["pair_indices"] if "pair_indices" in data.files else None

    if outdir is None:
        outdir = make_output_dir(path)
    else:
        os.makedirs(outdir, exist_ok=True)

    timeseries, saturation_info = _apply_saturation_policy(timeseries_raw, cfg, outdir=outdir)
    kept_curve_indices = saturation_info.get("kept_curve_indices", None) if isinstance(saturation_info, dict) else None
    if pair_indices is not None and kept_curve_indices is not None:
        kept_curve_indices = np.asarray(kept_curve_indices, dtype=int)
        if kept_curve_indices.size > 0:
            pair_indices = np.asarray(pair_indices, dtype=int)[kept_curve_indices]
        else:
            pair_indices = np.asarray(pair_indices, dtype=int)

    res = _process_timeseries_array(timeseries, path, cfg, outdir, pair_indices=pair_indices, saturation_info=saturation_info)
    res["saturation"] = saturation_info
    # Save outputs
    save_results(outdir, "lyapunov_results", res)
    # Optionally save averaged curve
    if "average_curve" in res:
        save_npz(outdir, "lyapunov_average", average_curve=np.array(res["average_curve"]))

    if "lyapunov_time_series" in res:
        lyap_arr = np.asarray(res["lyapunov_time_series"], dtype=float)
        dist_arr = np.asarray(timeseries, dtype=float)
        t = np.arange(lyap_arr.shape[1], dtype=int)
        save_kwargs = {
            "lambda_t": lyap_arr,
            "distance_timeseries": dist_arr,
            "time": t,
        }
        if pair_indices is not None:
            save_kwargs["pair_indices"] = np.asarray(pair_indices, dtype=int)
        save_npz(
            outdir,
            "lyapunov_time_series",
            **save_kwargs,
        )

        if cfg.get("time_dependent", {}).get("save_csv", True):
            _save_ftle_csv(outdir, "lyapunov_time_series.csv", lyap_arr, dist_arr, pair_indices=pair_indices)

    return res


def _apply_saturation_policy(timeseries: np.ndarray, cfg: LyapunovConfig, outdir: str = None):
    arr = np.asarray(timeseries, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"timeseries must be 2D (n_pairs, T), got shape={arr.shape}")

    sat_cfg = cfg.get("exclude_saturation", {}) or {}
    mode = str(sat_cfg.get("mode", "none")).lower()

    n_pairs, T = arr.shape
    info = {
        "mode": mode,
        "original_length": int(T),
        "used_length": int(T),
        "saturation_index": None,
        "jump_index": None,
        "baseline": None,
        "plateau": None,
        "midpoint": None,
        "smooth_window": None,
    }

    if mode == "none" or T <= 2:
        return arr, info

    min_points = int(sat_cfg.get("min_points", 5))
    debug_plot = bool(sat_cfg.get("debug_plot", False))
    save_midpoint_distribution = bool(sat_cfg.get("save_midpoint_distribution", False))

    log_plot = bool(cfg.get("plot", {}).get("log_plot", False))

    arr_masked = np.array(arr, copy=True)
    per_curve = []

    for curve_idx in range(arr.shape[0]):
        curve = arr[curve_idx]
        det = detect_curve_transition_points(curve, sat_cfg=sat_cfg, total_length=T)
        jump_idx = int(det["jump_index"])
        sat_idx = int(det["saturation_index"])
        second_jump_idx = det.get("second_jump_index", None)

        if mode == "exclude_full":
            cutoff = sat_idx
        elif mode == "exclude_half":
            cutoff = sat_idx + (T - sat_idx) // 2
        else:
            cutoff = T

        cutoff = max(min_points, min(T, int(cutoff)))
        if cutoff < T:
            arr_masked[curve_idx, cutoff:] = np.nan

        baseline_plot = float(det["baseline_plot"])
        plateau_plot = float(det["plateau_plot"])
        midpoint_plot = float(det["midpoint_plot"])
        smooth_plot = np.asarray(det["smooth_curve_plot"], dtype=float)

        per_curve.append({
            "curve_index": int(curve_idx),
            "used_length": int(cutoff),
            "saturation_index": int(sat_idx),
            "jump_index": int(jump_idx),
            "second_jump_index": int(second_jump_idx) if second_jump_idx is not None else None,
            "baseline": float(baseline_plot),
            "plateau": float(plateau_plot),
            "midpoint": float(midpoint_plot),
            "smooth_window": int(det["smooth_window"]),
            "detect_in_log_scale": bool(det["detect_in_log_scale"]),
        })

        if outdir is not None and cfg.get("plot", {}).get("save", True) and debug_plot:
            try:
                times = np.arange(T, dtype=int)
                plot_saturation_detection(
                    times=times,
                    mean_curve=curve,
                    smooth_curve=smooth_plot,
                    jump_idx=int(jump_idx),
                    sat_idx=int(sat_idx),
                    baseline=float(baseline_plot),
                    plateau=float(plateau_plot),
                    midpoint=float(midpoint_plot),
                    outpath=os.path.join(outdir, f"saturation_detection_debug_curve_{curve_idx:04d}.png"),
                    log_plot=log_plot,
                )
            except Exception:
                pass

    used_lengths = [x["used_length"] for x in per_curve] if per_curve else [T]
    sat_indices = [x["saturation_index"] for x in per_curve] if per_curve else []
    jump_indices = [x["jump_index"] for x in per_curve] if per_curve else []
    midpoint_indices = [x["saturation_index"] for x in per_curve] if per_curve else []

    outlier_cfg = sat_cfg.get("midpoint_outlier_filter", {}) or {}
    outlier_enabled = bool(outlier_cfg.get("enabled", False))
    outlier_bins = int(outlier_cfg.get("num_bins", min(25, max(5, T // 4))))
    outlier_bins = max(3, min(outlier_bins, max(3, T)))
    neighbor_bins = int(outlier_cfg.get("neighbor_bins", 1))
    neighbor_bins = max(0, neighbor_bins)

    keep_mask = np.ones(arr_masked.shape[0], dtype=bool)
    selected_bin = None
    selected_bin_range = None
    midpoint_indices_filtered = list(midpoint_indices)

    if outlier_enabled and len(midpoint_indices) > 0:
        mids = np.asarray(midpoint_indices, dtype=float)
        mids = mids[np.isfinite(mids)]
        if mids.size > 0:
            hist, edges = np.histogram(mids, bins=outlier_bins, range=(0, T))
            if np.any(hist > 0):
                peak_bin = int(np.argmax(hist))
                left_bin = max(0, peak_bin - neighbor_bins)
                right_bin = min(outlier_bins - 1, peak_bin + neighbor_bins)
                left_edge = float(edges[left_bin])
                right_edge = float(edges[right_bin + 1])

                keep_mask = (
                    np.asarray(midpoint_indices, dtype=float) >= left_edge
                ) & (
                    np.asarray(midpoint_indices, dtype=float) < right_edge
                )
                if right_bin == outlier_bins - 1:
                    keep_mask = keep_mask | (np.asarray(midpoint_indices, dtype=float) == right_edge)

                if np.any(keep_mask):
                    arr_masked = arr_masked[keep_mask]
                    per_curve = [pc for pc, keep in zip(per_curve, keep_mask) if keep]
                    midpoint_indices_filtered = [int(x) for x, keep in zip(midpoint_indices, keep_mask) if keep]
                    used_lengths = [x["used_length"] for x in per_curve] if per_curve else [T]
                    sat_indices = [x["saturation_index"] for x in per_curve] if per_curve else []
                    jump_indices = [x["jump_index"] for x in per_curve] if per_curve else []
                    selected_bin = int(peak_bin)
                    selected_bin_range = [int(left_bin), int(right_bin)]
                else:
                    keep_mask = np.ones(arr_masked.shape[0], dtype=bool)

    if outdir is not None and save_midpoint_distribution and len(midpoint_indices) > 0:
        try:
            np.savez(
                os.path.join(outdir, "midpoint_index_distribution.npz"),
                midpoint_indices=np.asarray(midpoint_indices, dtype=int),
                midpoint_indices_filtered=np.asarray(midpoint_indices_filtered, dtype=int),
                kept_curve_indices=np.where(keep_mask)[0].astype(int),
            )
            plot_midpoint_distribution(
                midpoint_indices=np.asarray(midpoint_indices, dtype=int),
                outpath=os.path.join(outdir, "midpoint_index_distribution.png"),
            )
            if outlier_enabled and len(midpoint_indices_filtered) > 0:
                plot_midpoint_distribution(
                    midpoint_indices=np.asarray(midpoint_indices_filtered, dtype=int),
                    outpath=os.path.join(outdir, "midpoint_index_distribution_filtered.png"),
                )
        except Exception:
            pass

    info.update({
        "used_length": int(np.nanmin(used_lengths)) if len(used_lengths) > 0 else int(T),
        "saturation_index": int(np.nanmedian(sat_indices)) if len(sat_indices) > 0 else None,
        "jump_index": int(np.nanmedian(jump_indices)) if len(jump_indices) > 0 else None,
        "baseline": None,
        "plateau": None,
        "midpoint": None,
        "smooth_window": int(per_curve[0]["smooth_window"]) if len(per_curve) > 0 else None,
        "midpoint_indices": midpoint_indices_filtered,
        "midpoint_indices_raw": midpoint_indices,
        "kept_curve_indices": np.where(keep_mask)[0].astype(int).tolist(),
        "midpoint_outlier_filter": {
            "enabled": bool(outlier_enabled),
            "num_bins": int(outlier_bins),
            "neighbor_bins": int(neighbor_bins),
            "selected_peak_bin": selected_bin,
            "selected_bin_range": selected_bin_range,
            "n_kept": int(np.sum(keep_mask)),
            "n_total": int(len(keep_mask)),
        },
        "per_curve": per_curve,
    })

    return arr_masked, info


def process_path(path, cfg: LyapunovConfig = None):
    cfg = cfg or LyapunovConfig.from_global()
    outdir = make_output_dir(path)
    if os.path.isdir(path):
        files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".npz")]
        files.sort()
        results = {}
        for f in files:
            res = process_file(f, cfg)
            results[os.path.basename(f)] = res
        save_results(outdir, "batch_results", results)
        return results
    else:
        res = process_file(path, cfg)
        save_results(outdir, "results", res)
        return res


def _process_timeseries_array(timeseries, path, cfg: LyapunovConfig, outdir: str, pair_indices=None, saturation_info=None):
    # timeseries: shape (n_pairs, T)
    time_dep_cfg = cfg.get("time_dependent", {})
    if time_dep_cfg.get("enabled", False):
        return _process_time_dependent(timeseries, cfg, outdir, pair_indices=pair_indices, saturation_info=saturation_info)

    op = cfg.get("operation_mode", "fit_first")
    averaging_cfg = cfg.get("averaging", {})
    fit_cfg = cfg.get("fit", {})

    out = {}

    if op == "fit_first":
        fit_params = []
        n = timeseries.shape[0]
        save_plot = cfg.get("plot", {}).get("save", True)
        log_plot = cfg.get("plot", {}).get("log_plot", False)
        for i in range(n):
            s = timeseries[i]
            times = np.arange(len(s))
            skipped = detect_no_rise(s, averaging_cfg.get("outlier_detection", {}).get("max_baseline_frac", 0.1))
            res = None
            if not skipped:
                res = fit_timeseries(times, s, fitter_name=fit_cfg.get("type", "exponential"))
                fit_params.append(res)

            # plotting per-timeseries (either fitted or skipped)
            if save_plot:
                try:
                    fit_eval = None
                    if res is not None:
                        p = res.get("params")
                        if p is not None and len(p) >= 3:
                            a, lamb, c = p
                            fit_eval = a * np.exp(lamb * times) + c
                    filename = f"timeseries_{i:04d}"
                    if skipped:
                        filename += "_skipped"
                    plotpath = os.path.join(outdir, filename + ("_log.png" if log_plot else "_fit.png"))
                    plot_mean_and_fit(times, s, fit_eval, plotpath, log_plot=log_plot)
                except Exception:
                    pass

        out["fit_params"] = fit_params
        out["n_fitted"] = len(fit_params)
        # summary stats if numeric params available
        try:
            lambdas = [p.get("params")[1] for p in fit_params if p.get("params") is not None]
            out["lambda_mean"] = float(np.mean(lambdas)) if lambdas else None
            out["lambda_std"] = float(np.std(lambdas)) if lambdas else None
        except Exception:
            out["lambda_mean"] = None
            out["lambda_std"] = None

        # optional plotting of distribution
        if cfg.get("plot", {}).get("save", True):
            try:
                import matplotlib.pyplot as plt
                plt.hist([p.get("params")[1] for p in fit_params if p.get("params") is not None], bins=30)
                plt.title("Lyapunov lambdas distribution")
                histpath = os.path.join(outdir, "lambda_distribution.png")
                plt.savefig(histpath)
                plt.close()
                out["histogram"] = histpath
            except Exception:
                pass

    elif op == "average_first":
        curves = timeseries
        method = averaging_cfg.get("method", "aligned_mean")
        avg, aligned = average_curves(curves, method=method)
        times = np.arange(len(avg))
        res = fit_timeseries(times, avg, fitter_name=fit_cfg.get("type", "exponential"))
        out["average_curve"] = avg.tolist()
        out["fit_params"] = res
        # save plot of mean and fit if requested
        if cfg.get("plot", {}).get("save", True):
            fit_eval = None
            try:
                # construct fit curve if params available
                p = res.get("params")
                if p is not None:
                    a, lamb, c = p
                    fit_eval = a * np.exp(lamb * times) + c
            except Exception:
                fit_eval = None
            plotpath = os.path.join(outdir, "mean_and_fit.png")
            plot_mean_and_fit(times, avg, fit_eval, plotpath, log_plot=cfg.get("plot", {}).get("log_plot", False))
            out["plot"] = plotpath

    else:
        raise ValueError(f"Unknown operation mode: {op}")

    return out


def _compute_ftle_curves(timeseries: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(timeseries, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"timeseries must be 2D (n_pairs, T), got shape={arr.shape}")

    n_pairs, T = arr.shape
    d0 = np.maximum(arr[:, [0]], eps)
    arr_safe = np.maximum(arr, eps)

    ftle = np.full((n_pairs, T), np.nan, dtype=float)
    if T <= 1:
        return ftle

    t = np.arange(T, dtype=float)
    denom = np.maximum(t, 1.0)
    ftle[:, 1:] = np.log(arr_safe[:, 1:] / d0) / denom[1:]
    ftle[:, 0] = np.nan
    return ftle


def _align_curves_by_midpoint(curves: np.ndarray, midpoint_indices) -> np.ndarray:
    arr = np.asarray(curves, dtype=float)
    if arr.ndim != 2:
        return arr

    mids = np.asarray(midpoint_indices, dtype=float)
    if mids.size != arr.shape[0]:
        return arr

    valid = np.isfinite(mids)
    if not np.any(valid):
        return arr

    mids = np.round(mids).astype(int)
    ref = int(np.median(mids[valid]))
    max_shift_left = int(np.max(np.maximum(0, ref - mids[valid]))) if np.any(valid) else 0
    max_shift_right = int(np.max(np.maximum(0, mids[valid] - ref))) if np.any(valid) else 0
    T = arr.shape[1]
    out_T = T + max_shift_left + max_shift_right
    aligned = np.full((arr.shape[0], out_T), np.nan, dtype=float)

    for i in range(arr.shape[0]):
        if i >= mids.size or not np.isfinite(mids[i]):
            continue
        shift = int(ref - mids[i])
        start = max_shift_left + shift
        end = start + T
        s = max(0, start)
        e = min(out_T, end)
        src_s = s - start
        src_e = src_s + (e - s)
        if e > s:
            aligned[i, s:e] = arr[i, src_s:src_e]

    return aligned


def _save_ftle_csv(outdir: str, filename: str, lyap_arr: np.ndarray, dist_arr: np.ndarray, pair_indices: np.ndarray = None):
    csv_path = os.path.join(outdir, filename)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["pair_index", "traj_i", "traj_j", "time_index", "lambda_t", "distance"])
        n_pairs, T = lyap_arr.shape
        for pair_idx in range(n_pairs):
            if pair_indices is not None and pair_idx < len(pair_indices):
                traj_i = int(pair_indices[pair_idx][0])
                traj_j = int(pair_indices[pair_idx][1])
            else:
                traj_i = -1
                traj_j = -1
            for t in range(T):
                writer.writerow([
                    int(pair_idx),
                    traj_i,
                    traj_j,
                    int(t),
                    float(lyap_arr[pair_idx, t]),
                    float(dist_arr[pair_idx, t]),
                ])
    return csv_path


def _process_time_dependent(timeseries, cfg: LyapunovConfig, outdir: str, pair_indices: np.ndarray = None, saturation_info=None):
    td_cfg = cfg.get("time_dependent", {})
    mode = td_cfg.get("mode", "ftle")
    eps = float(td_cfg.get("eps", 1e-12))

    if mode != "ftle":
        raise ValueError(f"Unsupported time-dependent Lyapunov mode: {mode}")

    sat_info = saturation_info if isinstance(saturation_info, dict) else {}
    midpoint_indices = sat_info.get("midpoint_indices", []) if isinstance(sat_info, dict) else []

    align_distance = bool(td_cfg.get("align_distance_on_midpoint", False))
    align_lyap = bool(td_cfg.get("align_lyapunov_on_midpoint", False))

    distance_for_mean = np.asarray(timeseries, dtype=float)
    if align_distance and len(midpoint_indices) > 0:
        distance_for_mean = _align_curves_by_midpoint(distance_for_mean, midpoint_indices)

    lyap_ts = _compute_ftle_curves(timeseries, eps=eps)
    lyap_for_mean = lyap_ts
    if align_lyap and len(midpoint_indices) > 0:
        lyap_for_mean = _align_curves_by_midpoint(lyap_ts, midpoint_indices)

    mean_curve = np.nanmean(lyap_for_mean, axis=0)
    std_curve = np.nanstd(lyap_for_mean, axis=0)
    times = np.arange(mean_curve.shape[0], dtype=int)

    out = {
        "mode": "time_dependent",
        "time_dependent_mode": mode,
        "lyapunov_time_series": lyap_ts.tolist(),
        "mean_lyapunov_time_series": mean_curve.tolist(),
        "std_lyapunov_time_series": std_curve.tolist(),
        "distance_mean_aligned_on_midpoint": align_distance,
        "lyapunov_mean_aligned_on_midpoint": align_lyap,
        "n_pairs": int(lyap_ts.shape[0]),
        "time_length": int(mean_curve.shape[0]),
    }

    plot_cfg = cfg.get("plot", {})
    if plot_cfg.get("save", True):
        log_plot = bool(plot_cfg.get("log_plot", False))
        plot_lyapunov_time_series(
            times=times,
            mean_lambda=mean_curve,
            std_lambda=std_curve,
            outpath=os.path.join(outdir, "lyapunov_time_series_mean.png"),
            log_plot=log_plot,
            title=None,
        )

        if bool(plot_cfg.get("save_source_distance_timeseries", True)):
            source_metric = str(cfg.get("source_metric", "cos"))
            mean_dist = np.nanmean(distance_for_mean, axis=0)
            std_dist = np.nanstd(distance_for_mean, axis=0)
            plot_distance_time_series(
                times=np.arange(mean_dist.shape[0], dtype=int),
                mean_distance=mean_dist,
                std_distance=std_dist,
                outpath=os.path.join(outdir, f"{source_metric}_distance_time_series_mean.png"),
                log_plot=log_plot,
                title=None,
                ylabel=f"{source_metric} distance",
            )

        # Plot selected pairs (from config) while averaging still uses all pairs.
        pairs_to_plot = CONFIG.get("pairwise", {}).get("pairs_to_plot", []) or []
        if pair_indices is not None:
            pair_map = {
                (int(p[0]), int(p[1])): idx
                for idx, p in enumerate(np.asarray(pair_indices, dtype=int))
            }
        else:
            pair_map = {}

        for pair in pairs_to_plot:
            try:
                i, j = int(pair[0]), int(pair[1])
            except Exception:
                continue

            row_idx = None
            if pair_map:
                row_idx = pair_map.get((i, j), pair_map.get((j, i)))
            else:
                # Fallback: assume timeseries rows follow pairs_to_plot order.
                try:
                    row_idx = pairs_to_plot.index([i, j])
                except Exception:
                    row_idx = None

            if row_idx is None:
                continue
            if row_idx < 0 or row_idx >= lyap_ts.shape[0]:
                continue

            curve = lyap_ts[row_idx]
            zero_std = np.zeros_like(curve)
            plot_lyapunov_time_series(
                times=np.arange(curve.shape[0], dtype=int),
                mean_lambda=curve,
                std_lambda=zero_std,
                outpath=os.path.join(outdir, f"lyapunov_time_series_pair_{i}_{j}.png"),
                log_plot=log_plot,
                title=None,
            )

            if bool(plot_cfg.get("save_source_distance_timeseries", True)):
                dist_curve = np.asarray(timeseries[row_idx], dtype=float)
                plot_distance_time_series(
                    times=np.arange(dist_curve.shape[0], dtype=int),
                    mean_distance=dist_curve,
                    std_distance=np.zeros_like(dist_curve),
                    outpath=os.path.join(outdir, f"{source_metric}_distance_time_series_pair_{i}_{j}.png"),
                    log_plot=log_plot,
                    title=None,
                    ylabel=f"{source_metric} distance",
                )

    return out
