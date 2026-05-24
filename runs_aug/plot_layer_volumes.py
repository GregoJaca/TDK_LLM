import os
from pathlib import Path
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.config import Analysis, Default

# Global settings holding configured variables
SHOW_TITLE = False
DPI = 300

def find_volume_files(base_name: str):
    base_no_ext = os.path.splitext(base_name)[0]
    p = Path('.')
    return list(p.rglob(f"{base_no_ext}_*.pt"))


def safe_index(idx, N):
    if idx < 0:
        idx = N + idx
    return idx


def plot_pairs(vol_matrix, layers, pairs, outpath, method):
    # vol_matrix: (n_layers, N_traj)
    n_layers, N = vol_matrix.shape
    x = np.array(layers)
    plt.figure(figsize=(10, 6))
    cmap = plt.get_cmap('tab10')
    color_i = 0
    plotted_curves = 0
    for pair in pairs:
        a, b = pair
        ai = safe_index(a, N)
        bi = safe_index(b, N)
        if not (0 <= ai < N and 0 <= bi < N):
            continue
        plt.plot(x, vol_matrix[:, ai], label=f'traj_{a}', color=cmap(color_i % 10))
        plt.plot(x, vol_matrix[:, bi], label=f'traj_{b}', linestyle='--', color=cmap((color_i + 1) % 10))
        color_i += 2
        plotted_curves += 2
        
    plt.xlabel('Layer')
    plt.ylabel('Volume')
    if SHOW_TITLE:
        plt.title(f'Layer volumes - pairs - {method}')
    
    # Hide legend if only a single curve is plotted
    if plotted_curves > 1:
        plt.legend()
    plt.xticks(x)
    plt.tight_layout()
    plt.savefig(outpath, dpi=DPI)
    plt.close()


def plot_all_with_mean(vol_matrix, layers, outpath, method):
    n_layers, N = vol_matrix.shape
    x = np.array(layers)
    plt.figure(figsize=(10, 6))
    for i in range(N):
        plt.plot(x, vol_matrix[:, i], color='C0', alpha=0.12)
    mean = vol_matrix.mean(axis=1)
    plt.plot(x, mean, color='black', linewidth=2.0, label='mean')
    plt.xlabel('Layer')
    plt.ylabel('Volume')
    if SHOW_TITLE:
        plt.title(f'Layer volumes - all trajectories + mean - {method}')
    plt.legend()
    plt.xticks(x)
    plt.tight_layout()
    plt.savefig(outpath, dpi=DPI)
    plt.close()


def process_file(path: Path):
    fname = path.name
    base = os.path.splitext(Analysis.LAYER_VOLUME_OUTPUT_FILENAME)[0]
    # expect filename like base_method.pt
    if not fname.startswith(base + '_'):
        return
    method = fname[len(base) + 1:]
    method = os.path.splitext(method)[0]

    data = torch.load(path, map_location='cpu')
    layers = sorted([int(k) for k in data.keys()])
    # build matrix (n_layers, N)
    vol_list = []
    for L in layers:
        entry = data[int(L)]
        vols = entry['volumes']
        if isinstance(vols, torch.Tensor):
            vols = vols.numpy()
        vol_list.append(vols)
    vol_matrix = np.stack(vol_list, axis=0)

    pairs = getattr(Analysis, 'PAIRS_TO_PLOT', [])
    outdir = path.parent
    pairs_out = outdir / f"{base}_{method}_pairs.png"
    all_out = outdir / f"{base}_{method}_all.png"
    plot_pairs(vol_matrix, layers, pairs, str(pairs_out), method)
    plot_all_with_mean(vol_matrix, layers, str(all_out), method)


def configure_style():
    global SHOW_TITLE, DPI
    font_size = 14
    label_size = 14
    tick_size = 12
    legend_size = 11
    
    if os.path.exists("config.yaml"):
        try:
            import yaml
            with open("config.yaml", "r") as f:
                cfg = yaml.safe_load(f)
                plot_cfg = cfg.get("plotting", {})
                SHOW_TITLE = plot_cfg.get("show_title", SHOW_TITLE)
                font_size = plot_cfg.get("font_size", font_size)
                label_size = plot_cfg.get("label_size", label_size)
                tick_size = plot_cfg.get("tick_size", tick_size)
                legend_size = plot_cfg.get("legend_size", legend_size)
                DPI = plot_cfg.get("dpi", DPI)
        except Exception:
            pass
            
    plt.rcParams.update({
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "font.size": font_size,
        "axes.labelsize": label_size,
        "xtick.labelsize": tick_size,
        "ytick.labelsize": tick_size,
        "legend.fontsize": legend_size,
        "font.family": "DejaVu Serif",
    })


def main():
    configure_style()
    base = Analysis.LAYER_VOLUME_OUTPUT_FILENAME
    files = find_volume_files(base)
    for f in files:
        try:
            process_file(f)
        except Exception as e:
            print(f"Failed processing {f}: {e}")


if __name__ == '__main__':
    main()
