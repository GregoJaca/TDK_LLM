import numpy as np


def detect_no_rise(series, max_baseline_frac=0.1):
    baseline = np.median(series[: max(1, int(len(series) * 0.05))])
    peak = np.max(series)
    if peak <= baseline * (1 + max_baseline_frac):
        return True
    return False


def find_rise_index(series, threshold_rel=0.2):
    peak = np.max(series)
    thresh = peak * threshold_rel
    for i in range(len(series)):
        if series[i] >= thresh:
            return i
    return 0


def aligned_mean(curves, rise_index_fn=find_rise_index, truncate_saturation=True):
    # curves: (n_curves, T)
    rises = [rise_index_fn(c) for c in curves]
    min_post = min(len(c) - r for c, r in zip(curves, rises))
    aligned = np.array([c[r : r + min_post] for c, r in zip(curves, rises)])
    mean_curve = np.mean(aligned, axis=0)
    return mean_curve, aligned


def average_curves(curves, method="aligned_mean", cfg=None):
    if method == "mean":
        return np.mean(curves, axis=0), curves
    if method == "median":
        return np.median(curves, axis=0), curves
    if method == "aligned_mean":
        return aligned_mean(curves)
    raise ValueError(f"Unknown averaging method: {method}")
