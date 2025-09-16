import os
import numpy as np
from .config import LyapunovConfig
from .averaging import average_curves, detect_no_rise
from .fitters import fit_timeseries
from .plotting import plot_curves, plot_mean_and_fit
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
