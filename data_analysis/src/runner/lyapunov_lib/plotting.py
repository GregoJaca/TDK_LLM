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
