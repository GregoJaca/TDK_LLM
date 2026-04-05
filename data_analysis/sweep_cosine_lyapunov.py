"""
Sweep for cosine/Lyapunov robustness analysis.

Sweeps:
- CONFIG['sliding_window']['window_size']
- CONFIG['lyapunov']['exclude_saturation']['mode']
- CONFIG['EMBEDDING_CONFIG']['input_template']

For each combo, runs run_all.main and collects only:
- mean cosine distance time series
- mean Lyapunov time series
- one representative saturation debug image

Outputs are flattened into a single directory.
"""

import argparse
import copy
import csv
import json
import os
import re
import shutil
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from config import CONFIG
import run_all


def _as_list_of_ints(text: str) -> List[int]:
    vals = []
    for token in text.split(','):
        token = token.strip()
        if not token:
            continue
        vals.append(int(token))
    return vals


def _as_list_of_strings(text: str) -> List[str]:
    vals = []
    for token in text.split(','):
        token = token.strip()
        if token:
            vals.append(token)
    return vals


def _load_run_curves(run_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    lyap_npz = os.path.join(run_dir, 'lyapunov_time_series.npz')
    if not os.path.exists(lyap_npz):
        raise FileNotFoundError(f"Expected file not found: {lyap_npz}")

    data = np.load(lyap_npz)
    lambda_t = np.asarray(data['lambda_t'], dtype=float)
    distance_ts = np.asarray(data['distance_timeseries'], dtype=float)

    mean_lambda = np.nanmean(lambda_t, axis=0)
    mean_distance = np.nanmean(distance_ts, axis=0)
    time = np.arange(mean_lambda.shape[0], dtype=int)
    return time, mean_distance, mean_lambda


def _plot_multi_curve(curves: Dict[str, np.ndarray], outpath: str, ylabel: str, log_plot: bool = False):
    plt.figure(figsize=(8, 5))
    for label, values in curves.items():
        x = np.arange(len(values), dtype=int)
        plt.plot(x, values, linewidth=1.2, alpha=0.85, label=label)

    if log_plot:
        arr = np.concatenate([np.asarray(v, dtype=float) for v in curves.values()])
        if np.nanmin(arr) > 0:
            plt.yscale('log')
        else:
            plt.yscale('symlog', linthresh=1e-6)

    plt.xlabel('Time index', fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.grid(alpha=0.25)
    plt.legend(fontsize=9, ncol=2)
    plt.tight_layout()

    formats = CONFIG.get('plots', {}).get('output_formats', ['png'])
    if not isinstance(formats, (list, tuple)) or len(formats) == 0:
        formats = ['png']

    base, _ = os.path.splitext(outpath)
    for fmt in formats:
        fmt_l = str(fmt).lower().strip('.')
        if fmt_l in ('png', 'pdf'):
            plt.savefig(f"{base}.{fmt_l}", dpi=300)
    plt.close()


def _copy_if_exists(src: str, dst: str):
    if os.path.exists(src):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)


def _sanitize_template_name(template: str) -> str:
    name = template.replace('.pt', '')
    name = name.replace('{i}', 'i')
    name = re.sub(r'[^A-Za-z0-9_\-]+', '_', name)
    return name.strip('_')


def flatten_selected_timeseries(out_dir: str, flat_dir_name: str = "flat_selected_timeseries") -> str:
    """Collect selected timeseries/debug plots from combo subfolders into one flat folder.

    Expected combo folder name format:
    window_<ws>__mode_<mode>__lyapalign_<true|false>__template_<template_tag>
    """
    flat_dir = os.path.join(out_dir, flat_dir_name)
    os.makedirs(flat_dir, exist_ok=True)

    combo_re = re.compile(r"^window_(?P<ws>\d+)__mode_(?P<mode>.+?)__lyapalign_(?P<align>true|false)__template_(?P<tag>.+)$")
    formats = CONFIG.get('plots', {}).get('output_formats', ['png'])
    if not isinstance(formats, (list, tuple)) or len(formats) == 0:
        formats = ['png']
    formats = [str(fmt).lower().strip('.') for fmt in formats if str(fmt).lower().strip('.') in ('png', 'pdf')]
    if len(formats) == 0:
        formats = ['png']

    for name in sorted(os.listdir(out_dir)):
        combo_path = os.path.join(out_dir, name)
        if not os.path.isdir(combo_path):
            continue

        match = combo_re.match(name)
        if not match:
            continue

        ws = match.group('ws')
        mode = match.group('mode')
        align = match.group('align')
        tag = match.group('tag')
        run_dir = os.path.join(combo_path, 'sweep')
        if not os.path.isdir(run_dir):
            continue

        for fmt in formats:
            src_lyap = os.path.join(run_dir, f"lyapunov_time_series_mean.{fmt}")
            src_cos = os.path.join(run_dir, f"cos_distance_time_series_mean.{fmt}")
            src_dbg = os.path.join(run_dir, f"saturation_detection_debug_curve_0000.{fmt}")

            dst_lyap = os.path.join(flat_dir, f"lyapunov_time_series_mean_{ws}_{mode}_{align}_{tag}.{fmt}")
            dst_cos = os.path.join(flat_dir, f"cos_distance_time_series_mean_{ws}_{mode}_{align}_{tag}.{fmt}")
            dst_dbg = os.path.join(flat_dir, f"saturation_debug_{ws}_{mode}_{align}_{tag}.{fmt}")

            _copy_if_exists(src_lyap, dst_lyap)
            _copy_if_exists(src_cos, dst_cos)
            _copy_if_exists(src_dbg, dst_dbg)

    return flat_dir


def run_sweep(
    input_path: str,
    out_dir: str,
    window_sizes: List[int],
    saturation_modes: List[str],
    input_templates: List[str],
    align_lyap_options: List[bool],
):
    os.makedirs(out_dir, exist_ok=True)

    original = copy.deepcopy(CONFIG)

    # Keep plotting focused during sweep.
    CONFIG.setdefault('plots', {})['save_histograms'] = False
    CONFIG.setdefault('metrics', {})['save_plots'] = False
    CONFIG.setdefault('lyapunov', {}).setdefault('plot', {})['save'] = True
    CONFIG['lyapunov']['plot']['save_source_distance_timeseries'] = True
    CONFIG.setdefault('lyapunov', {}).setdefault('exclude_saturation', {})['debug_plot'] = True

    summary_rows = []
    cosine_curves = {}
    lyap_curves = {}

    for ws in window_sizes:
        for sat_mode in saturation_modes:
            for input_template in input_templates:
                for align_lyap in align_lyap_options:
                    template_tag = _sanitize_template_name(input_template)
                    align_tag = "true" if align_lyap else "false"
                    combo_name = f"window_{ws}__mode_{sat_mode}__lyapalign_{align_tag}__template_{template_tag}"
                    combo_root = os.path.join(out_dir, combo_name)
                    os.makedirs(combo_root, exist_ok=True)

                    CONFIG['sliding_window']['window_size'] = ws
                    CONFIG['lyapunov']['exclude_saturation']['mode'] = sat_mode
                    CONFIG['lyapunov']['time_dependent']['align_lyapunov_on_midpoint'] = bool(align_lyap)
                    CONFIG.setdefault('EMBEDDING_CONFIG', {})['input_template'] = input_template

                    # Use sweep mode in run_all so output is deterministic under combo_root/sweep
                    run_all.main(input_path=input_path, results_root=combo_root, sweep_param_value=ws)

                    run_dir = os.path.join(combo_root, 'sweep')
                    t, mean_cos, mean_lyap = _load_run_curves(run_dir)

                    cosine_curves[combo_name] = mean_cos
                    lyap_curves[combo_name] = mean_lyap

                    np.savez(
                        os.path.join(out_dir, f"{combo_name}_mean_curves.npz"),
                        time=t,
                        mean_cos=mean_cos,
                        mean_lyap=mean_lyap,
                    )

                    summary_rows.append({
                        'combo': combo_name,
                        'window_size': ws,
                        'saturation_mode': sat_mode,
                        'align_lyapunov_on_midpoint': bool(align_lyap),
                        'input_template': input_template,
                        'mean_cos_over_time': float(np.nanmean(mean_cos)),
                        'mean_lyap_over_time': float(np.nanmean(mean_lyap)),
                    })

    # Combined plots
    _plot_multi_curve(
        curves=cosine_curves,
        outpath=os.path.join(out_dir, 'mean_cosine_timeseries_sweep.png'),
        ylabel='Mean cosine distance',
        log_plot=bool(CONFIG.get('plots', {}).get('log_plot', False)),
    )
    _plot_multi_curve(
        curves=lyap_curves,
        outpath=os.path.join(out_dir, 'mean_lyapunov_timeseries_sweep.png'),
        ylabel='Mean Lyapunov $\\lambda(t)$',
        log_plot=bool(CONFIG.get('lyapunov', {}).get('plot', {}).get('log_plot', False)),
    )

    # CSV summary
    csv_path = os.path.join(out_dir, 'sweep_summary.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=['combo', 'window_size', 'saturation_mode', 'align_lyapunov_on_midpoint', 'input_template', 'mean_cos_over_time', 'mean_lyap_over_time'],
        )
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    with open(os.path.join(out_dir, 'sweep_config_used.json'), 'w', encoding='utf-8') as f:
        json.dump(
            {
                'window_sizes': window_sizes,
                'saturation_modes': saturation_modes,
                'align_lyapunov_on_midpoint_options': align_lyap_options,
                'input_templates': input_templates,
                'input_path': input_path,
            },
            f,
            indent=2,
        )

    flat_dir = flatten_selected_timeseries(out_dir)
    print(f"Flattened selected plots to: {flat_dir}")

    # Restore original config in-memory
    CONFIG.clear()
    CONFIG.update(original)


def main():
    parser = argparse.ArgumentParser(description='Sweep: window size x saturation mode x input template')
    parser.add_argument('--input', required=True, help='Input .pt file or directory path used by run_all')
    parser.add_argument('--out', default='sweep_cos_lyapunov_results', help='Output directory for sweep artifacts')
    parser.add_argument('--window_sizes', default='1,4,16,64', help='Comma-separated ints, e.g. 1,4,16,64')
    parser.add_argument('--saturation_modes', default='exclude_half,none,exclude_full', help='Comma-separated values from none,exclude_half,exclude_full')
    parser.add_argument('--input_templates', default='hidden_states_cond_{i}_layer_-1.pt,hidden_states_cond_{i}_layer_22.pt', help='Comma-separated template strings')
    parser.add_argument('--align_lyap_options', default='false,true', help='Comma-separated booleans for align_lyapunov_on_midpoint, e.g. false,true')
    parser.add_argument('--postprocess_only', action='store_true', help='Only flatten existing combo outputs into one folder')
    args = parser.parse_args()

    if args.postprocess_only:
        flat_dir = flatten_selected_timeseries(args.out)
        print(f"Flattened selected plots to: {flat_dir}")
        return

    window_sizes = _as_list_of_ints(args.window_sizes)
    saturation_modes = _as_list_of_strings(args.saturation_modes)
    input_templates = _as_list_of_strings(args.input_templates)
    align_lyap_options = [s.strip().lower() in ('1', 'true', 'yes', 'y') for s in _as_list_of_strings(args.align_lyap_options)]

    run_sweep(
        input_path=args.input,
        out_dir=args.out,
        window_sizes=window_sizes,
        saturation_modes=saturation_modes,
        input_templates=input_templates,
        align_lyap_options=align_lyap_options,
    )


if __name__ == '__main__':
    main()
