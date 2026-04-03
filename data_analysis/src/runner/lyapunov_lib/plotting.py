import matplotlib.pyplot as plt
import numpy as np
import os


def plot_curves(times, curves, outpath, labels=None, log_plot=False, title=None):
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.figure(figsize=(6, 4))
    for i, c in enumerate(curves):
        lab = None if labels is None else labels[i]
        if log_plot:
            plt.plot(times, np.log(c + 1e-12), label=lab, alpha=0.7)
        else:
            plt.plot(times, c, label=lab, alpha=0.7)
    if labels is not None:
        plt.legend(fontsize=16)
    # if title:
    #     plt.title(title)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.savefig(outpath.replace('.png', '.pdf'), dpi=300)
    plt.close()


def plot_mean_and_fit(times, mean_curve, fit_curve, outpath, log_plot=False, title=None):
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.figure(figsize=(6, 4))
    if log_plot:
        plt.plot(times, np.log(mean_curve + 1e-12), label="mean")
        if fit_curve is not None:
            plt.plot(times, np.log(fit_curve + 1e-12), label="fit")
    else:
        plt.plot(times, mean_curve, label="mean")
        if fit_curve is not None:
            plt.plot(times, fit_curve, label="fit")
    plt.legend(fontsize=16)
    # if title:
    #     plt.title(title)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.savefig(outpath.replace('.png', '.pdf'), dpi=300)
    plt.close()


def plot_lyapunov_time_series(times, mean_lambda, std_lambda, outpath, log_plot=False, title=None):
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.figure(figsize=(6, 4))

    mean_lambda = np.asarray(mean_lambda, dtype=float)
    std_lambda = np.asarray(std_lambda, dtype=float)
    lower = mean_lambda - std_lambda
    upper = mean_lambda + std_lambda

    plt.plot(times, mean_lambda, label="mean $\\lambda(t)$", linewidth=1.5)
    plt.fill_between(times, lower, upper, alpha=0.2, label="±1 std")

    if log_plot:
        # symlog supports negative values while still providing log-like scaling.
        plt.yscale("symlog", linthresh=1e-4)

    if title:
        plt.title(title)

    plt.xlabel("Time index", fontsize=16)
    plt.ylabel("$\\lambda(t)$", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.25)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.savefig(outpath.replace('.png', '.pdf'), dpi=300)
    plt.close()


def plot_saturation_detection(
    times,
    mean_curve,
    smooth_curve,
    jump_idx,
    sat_idx,
    baseline,
    plateau,
    midpoint,
    outpath,
    log_plot=False,
):
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.figure(figsize=(8, 4))

    plt.plot(times, mean_curve, label="mean distance", linewidth=1.2, alpha=0.7)
    plt.plot(times, smooth_curve, label="smoothed mean", linewidth=1.8)

    plt.axvline(jump_idx, color="tab:orange", linestyle="--", alpha=0.8, label=f"jump idx={jump_idx}")
    plt.axvline(sat_idx, color="tab:red", linestyle="-.", alpha=0.9, label=f"sat idx={sat_idx}")

    plt.axhline(baseline, color="tab:green", linestyle=":", alpha=0.8, label="baseline")
    plt.axhline(plateau, color="tab:purple", linestyle=":", alpha=0.8, label="plateau")
    plt.axhline(midpoint, color="tab:brown", linestyle="--", alpha=0.9, label="midpoint")

    if log_plot:
        if np.nanmin(mean_curve) > 0 and np.nanmin(smooth_curve) > 0 and baseline > 0 and plateau > 0 and midpoint > 0:
            plt.yscale("log")
        else:
            plt.yscale("symlog", linthresh=1e-6)

    plt.xlabel("Time index", fontsize=16)
    plt.ylabel("Distance", fontsize=16)
    plt.legend(fontsize=10, ncol=2)
    plt.grid(alpha=0.25)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.savefig(outpath.replace('.png', '.pdf'), dpi=300)
    plt.close()
