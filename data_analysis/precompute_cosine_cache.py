import copy
import os
import shutil
import argparse

from config import CONFIG
import run_all


def _restore_config(snapshot):
    CONFIG.clear()
    CONFIG.update(snapshot)


def _set_only_metric_enabled(metric_name: str):
    metrics_cfg = CONFIG.get("metrics", {})
    for name, cfg in metrics_cfg.items():
        if isinstance(cfg, dict) and "enabled" in cfg:
            cfg["enabled"] = (name == metric_name)


def _ensure_parent_dir(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


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
    parser = argparse.ArgumentParser(description="Precompute and save cosine pairwise distances using current pipeline functions.")
    parser.add_argument("--input", type=str, default=defaults.get("input_path"), help="Input file/folder for run_all.py")
    parser.add_argument("--cosine_out_dir", type=str, default=defaults.get("cosine_out_dir", "cosine_distances"), help="Directory where run outputs cosine distances")
    parser.add_argument("--cosine_npz_path", type=str, default=defaults.get("cosine_npz_path", defaults.get("cached_npz_path")), help="Optional explicit path to copy cosine NPZ to")
    parser.add_argument("--metric_name", type=str, default=defaults.get("metric_name", "cos"), help="Metric to precompute")
    parser.add_argument("--force_recompute", type=_as_bool, default=bool(defaults.get("force_recompute", False)), help="Recompute even if output file already exists")
    return parser.parse_args()


def main():
    pipe = CONFIG.get("cached_cosine_pipeline", {})
    args = _build_args(pipe)
    input_path = args.input
    cosine_out_dir = args.cosine_out_dir
    explicit_npz_path = args.cosine_npz_path
    metric_name = args.metric_name
    force_recompute = bool(args.force_recompute)

    if not input_path:
        raise ValueError("CONFIG['cached_cosine_pipeline']['input_path'] must be set")

    target_npz = explicit_npz_path if explicit_npz_path else os.path.join(cosine_out_dir, f"{metric_name}_pairwise_timeseries.npz")
    if os.path.exists(target_npz) and not force_recompute:
        print(f"[precompute] cosine distances already exist, skipping compute: {target_npz}")
        if explicit_npz_path:
            _ensure_parent_dir(explicit_npz_path)
            print(f"[precompute] using explicit cosine NPZ path: {explicit_npz_path}")
        return

    os.makedirs(cosine_out_dir, exist_ok=True)

    snapshot = copy.deepcopy(CONFIG)
    try:
        pre_cfg = pipe.get("precompute", {})
        CONFIG.setdefault("pairwise", {})["save_pairwise_aggregated"] = True

        if bool(pre_cfg.get("only_source_metric", True)):
            _set_only_metric_enabled(metric_name)

        if bool(pre_cfg.get("disable_lyapunov", True)):
            CONFIG.setdefault("lyapunov", {})["enabled"] = False

        if bool(pre_cfg.get("disable_histograms", True)):
            CONFIG.setdefault("plots", {})["save_histograms"] = False

        if bool(pre_cfg.get("disable_metric_plots", True)):
            CONFIG.setdefault("metrics", {})["save_plots"] = False

        run_info = run_all.main(input_path=input_path, results_root=cosine_out_dir, sweep_param_value=None)
        if not isinstance(run_info, dict) or "results_dir" not in run_info:
            raise RuntimeError("run_all.main did not return results metadata as expected")

        produced_npz = os.path.join(run_info["results_dir"], f"{metric_name}_pairwise_timeseries.npz")

        if not os.path.exists(produced_npz):
            raise FileNotFoundError(
                f"Expected metric file not found after run: {produced_npz}"
            )

        _ensure_parent_dir(target_npz)
        shutil.copy2(produced_npz, target_npz)
        print(f"[precompute] cosine distances written to: {target_npz}")
    finally:
        _restore_config(snapshot)


if __name__ == "__main__":
    main()
