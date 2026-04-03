import os
import numpy as np
import csv
from .config import LyapunovConfig
from .averaging import average_curves, detect_no_rise
from .fitters import fit_timeseries
from .plotting import plot_curves, plot_mean_and_fit, plot_lyapunov_time_series
from .results import make_output_dir, save_results, save_npz
import json


def process_file(path, cfg: LyapunovConfig = None):
    cfg = cfg or LyapunovConfig.from_global()
    data = np.load(path)
    timeseries = data["timeseries"]
    outdir = make_output_dir(path)
    res = _process_timeseries_array(timeseries, path, cfg, outdir)
    # Save outputs
    save_results(outdir, os.path.splitext(os.path.basename(path))[0] + "_results", res)
    # Optionally save averaged curve
    if "average_curve" in res:
        save_npz(outdir, os.path.splitext(os.path.basename(path))[0] + "_average", average_curve=np.array(res["average_curve"]))

    if "lyapunov_time_series" in res:
        base_name = os.path.splitext(os.path.basename(path))[0]
        lyap_arr = np.asarray(res["lyapunov_time_series"], dtype=float)
        dist_arr = np.asarray(timeseries, dtype=float)
        t = np.arange(lyap_arr.shape[1], dtype=int)
        save_npz(
            outdir,
            f"{base_name}_lyapunov_time_series",
            lambda_t=lyap_arr,
            distance_timeseries=dist_arr,
            time=t,
        )

        if cfg.get("time_dependent", {}).get("save_csv", True):
            _save_ftle_csv(outdir, f"{base_name}_lyapunov_time_series.csv", lyap_arr, dist_arr)

    return res


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


def _process_timeseries_array(timeseries, path, cfg: LyapunovConfig, outdir: str):
    # timeseries: shape (n_pairs, T)
    time_dep_cfg = cfg.get("time_dependent", {})
    if time_dep_cfg.get("enabled", False):
        return _process_time_dependent(timeseries, cfg, outdir)

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
    ftle[:, 0] = 0.0
    return ftle


def _save_ftle_csv(outdir: str, filename: str, lyap_arr: np.ndarray, dist_arr: np.ndarray):
    csv_path = os.path.join(outdir, filename)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["pair_index", "time_index", "lambda_t", "distance"])
        n_pairs, T = lyap_arr.shape
        for pair_idx in range(n_pairs):
            for t in range(T):
                writer.writerow([
                    int(pair_idx),
                    int(t),
                    float(lyap_arr[pair_idx, t]),
                    float(dist_arr[pair_idx, t]),
                ])
    return csv_path


def _process_time_dependent(timeseries, cfg: LyapunovConfig, outdir: str):
    td_cfg = cfg.get("time_dependent", {})
    mode = td_cfg.get("mode", "ftle")
    eps = float(td_cfg.get("eps", 1e-12))

    if mode != "ftle":
        raise ValueError(f"Unsupported time-dependent Lyapunov mode: {mode}")

    lyap_ts = _compute_ftle_curves(timeseries, eps=eps)
    mean_curve = np.nanmean(lyap_ts, axis=0)
    std_curve = np.nanstd(lyap_ts, axis=0)
    times = np.arange(lyap_ts.shape[1], dtype=int)

    out = {
        "mode": "time_dependent",
        "time_dependent_mode": mode,
        "lyapunov_time_series": lyap_ts.tolist(),
        "mean_lyapunov_time_series": mean_curve.tolist(),
        "std_lyapunov_time_series": std_curve.tolist(),
        "n_pairs": int(lyap_ts.shape[0]),
        "time_length": int(lyap_ts.shape[1]),
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

    return out
