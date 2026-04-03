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
