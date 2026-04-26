import argparse
import copy
import json
import os

import numpy as np

from config import CONFIG
from src.runner.lyapunov_lib.config import LyapunovConfig
from src.runner.lyapunov_lib.core import detect_curve_transition_points
from src.runner.lyapunov_lib.plotting import plot_midpoint_distribution, plot_saturation_detection


def _restore_config(snapshot):
    CONFIG.clear()
    CONFIG.update(snapshot)


def _as_bool(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "y", "on"):
        return True
    if text in ("0", "false", "no", "n", "off"):
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def _build_args(defaults):
    midpoint_cfg = defaults.get("midpoint_analysis", {}) if isinstance(defaults, dict) else {}

    parser = argparse.ArgumentParser(
        description="Analyze first and second jump-time distributions from precomputed cosine distances."
    )
    parser.add_argument(
        "--cosine_npz_path",
        type=str,
        default=defaults.get("cosine_npz_path", "cosine_distances/cos_pairwise_timeseries.npz"),
        help="Path to cosine pairwise timeseries NPZ",
    )
    parser.add_argument(
        "--analysis_out_dir",
        type=str,
        default=midpoint_cfg.get("output_dir", "jump_analysis"),
        help="Directory for analysis outputs",
    )
    parser.add_argument(
        "--debug_plots_dir",
        type=str,
        default=midpoint_cfg.get("debug_plots_dir", "jump_analysis/debug_plots"),
        help="Directory for debug per-curve plots",
    )
    parser.add_argument(
        "--debug_plot",
        type=_as_bool,
        default=bool(midpoint_cfg.get("debug_plot", True)),
        help="Enable per-curve debug plots",
    )
    parser.add_argument(
        "--save_midpoint_distribution",
        type=_as_bool,
        default=bool(midpoint_cfg.get("save_midpoint_distribution", True)),
        help="Save first-jump midpoint distribution plot",
    )
    parser.add_argument(
        "--save_second_jump_distribution",
        type=_as_bool,
        default=bool(midpoint_cfg.get("save_second_jump_distribution", True)),
        help="Save second-jump distribution plot",
    )
    return parser.parse_args()


def main():
    pipe = CONFIG.get("cached_cosine_pipeline", {})
    args = _build_args(pipe)

    if not os.path.exists(args.cosine_npz_path):
        raise FileNotFoundError(f"Cosine NPZ not found: {args.cosine_npz_path}")

    os.makedirs(args.analysis_out_dir, exist_ok=True)
    os.makedirs(args.debug_plots_dir, exist_ok=True)

    data = np.load(args.cosine_npz_path)
    if "timeseries" not in data.files:
        raise ValueError(f"NPZ does not contain 'timeseries': {args.cosine_npz_path}")

    timeseries = np.asarray(data["timeseries"], dtype=float)
    pair_indices = np.asarray(data["pair_indices"], dtype=int) if "pair_indices" in data.files else None

    if timeseries.ndim != 2:
        raise ValueError(f"Expected timeseries shape (n_pairs, T), got {timeseries.shape}")

    cfg = LyapunovConfig.from_global()
    sat_cfg = cfg.get("exclude_saturation", {}) or {}
    log_plot = bool(cfg.get("plot", {}).get("log_plot", False))

    snapshot = copy.deepcopy(CONFIG)
    try:
        CONFIG.setdefault("lyapunov", {}).setdefault("exclude_saturation", {})["debug_plot"] = bool(args.debug_plot)

        first_midpoint_indices = []
        second_jump_indices = []
        per_curve = []

        for curve_idx in range(timeseries.shape[0]):
            curve = np.asarray(timeseries[curve_idx], dtype=float)
            det = detect_curve_transition_points(curve, sat_cfg=sat_cfg, total_length=curve.shape[0])

            first_midpoint = int(det["saturation_index"])
            second_jump = det.get("second_jump_index", None)

            first_midpoint_indices.append(first_midpoint)
            if second_jump is not None:
                second_jump_indices.append(int(second_jump))

            pair = None
            if pair_indices is not None and curve_idx < pair_indices.shape[0]:
                pair = [int(pair_indices[curve_idx, 0]), int(pair_indices[curve_idx, 1])]

            per_curve.append(
                {
                    "curve_index": int(curve_idx),
                    "pair": pair,
                    "first_midpoint_index": int(first_midpoint),
                    "first_jump_index": int(det["jump_index"]),
                    "second_jump_index": int(second_jump) if second_jump is not None else None,
                    "smooth_window": int(det["smooth_window"]),
                }
            )

            if args.debug_plot:
                plot_saturation_detection(
                    times=np.arange(curve.shape[0], dtype=int),
                    mean_curve=curve,
                    smooth_curve=np.asarray(det["smooth_curve_plot"], dtype=float),
                    jump_idx=int(det["jump_index"]),
                    sat_idx=int(det["saturation_index"]),
                    baseline=float(det["baseline_plot"]),
                    plateau=float(det["plateau_plot"]),
                    midpoint=float(det["midpoint_plot"]),
                    outpath=os.path.join(args.debug_plots_dir, f"jump_debug_curve_{curve_idx:04d}.png"),
                    log_plot=log_plot,
                )

        first_midpoint_arr = np.asarray(first_midpoint_indices, dtype=int)
        second_jump_arr = np.asarray(second_jump_indices, dtype=int)

        np.savez(
            os.path.join(args.analysis_out_dir, "jump_index_distributions.npz"),
            first_midpoint_indices=first_midpoint_arr,
            second_jump_indices=second_jump_arr,
        )

        if args.save_midpoint_distribution and first_midpoint_arr.size > 0:
            plot_midpoint_distribution(
                midpoint_indices=first_midpoint_arr,
                outpath=os.path.join(args.analysis_out_dir, "first_midpoint_index_distribution.png"),
            )

        if args.save_second_jump_distribution and second_jump_arr.size > 0:
            plot_midpoint_distribution(
                midpoint_indices=second_jump_arr,
                outpath=os.path.join(args.analysis_out_dir, "second_jump_index_distribution.png"),
            )

        summary = {
            "input_npz": args.cosine_npz_path,
            "n_curves": int(timeseries.shape[0]),
            "time_length": int(timeseries.shape[1]),
            "first_midpoint": {
                "count": int(first_midpoint_arr.size),
                "mean": float(np.nanmean(first_midpoint_arr)) if first_midpoint_arr.size > 0 else None,
                "std": float(np.nanstd(first_midpoint_arr)) if first_midpoint_arr.size > 0 else None,
            },
            "second_jump": {
                "count": int(second_jump_arr.size),
                "mean": float(np.nanmean(second_jump_arr)) if second_jump_arr.size > 0 else None,
                "std": float(np.nanstd(second_jump_arr)) if second_jump_arr.size > 0 else None,
            },
            "per_curve": per_curve,
        }

        with open(os.path.join(args.analysis_out_dir, "jump_analysis_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"[jump-analysis] completed. Outputs: {args.analysis_out_dir}")
        print(f"[jump-analysis] debug plots: {args.debug_plots_dir}")
    finally:
        _restore_config(snapshot)


if __name__ == "__main__":
    main()
