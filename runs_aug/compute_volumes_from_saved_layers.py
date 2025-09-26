import os
from pathlib import Path
import torch
from src.config import Default, Analysis
from src.analysis import compute_all_layers_volumes


def collect_hidden_states(results_dir: str, selected_layers, n_conditions):
    res = {}
    for layer_idx in selected_layers:
        layer_list = []
        for i in range(n_conditions):
            fname = os.path.join(results_dir, f"hidden_states_cond_{i}_layer_{layer_idx}.pt")
            if not os.path.exists(fname):
                raise FileNotFoundError(f"Missing hidden state file: {fname}")
            tensor = torch.load(fname, map_location='cpu')
            # tensor expected shape (T, D)
            layer_list.append(tensor)
        # stack into (N, T, D)
        res[int(layer_idx)] = torch.stack(layer_list, dim=0)
    return res


def main():
    base_results_dir = Path(Default.RESULTS_DIR)
    selected_layers = Default.SELECTED_LAYERS
    n_conditions = Default.N_INITIAL_CONDITIONS

    if not base_results_dir.exists():
        raise NotADirectoryError(f"Results dir not found: {base_results_dir}")

    # find run subdirectories that contain hidden_states_cond files
    run_dirs = []
    for root, dirs, files in os.walk(base_results_dir):
        for fname in files:
            if fname.startswith('hidden_states_cond_') and fname.endswith('.pt'):
                run_dirs.append(Path(root))
                break
    run_dirs = sorted(set(run_dirs))
    if not run_dirs:
        raise FileNotFoundError(f"No hidden_states_cond_*.pt files found under {base_results_dir}")

    methods = Analysis.LAYER_VOLUME_METHOD
    normalize = Analysis.NORMALIZE_HIDDEN_STATES
    n_axes = Analysis.LAYER_VOLUME_N_AXES
    base = Analysis.LAYER_VOLUME_OUTPUT_FILENAME
    base_no_ext, ext = os.path.splitext(base)
    if ext == '':
        ext = '.pt'

    for run_dir in run_dirs:
        try:
            hidden_states = collect_hidden_states(str(run_dir), selected_layers, n_conditions)
        except FileNotFoundError as e:
            print(f"Skipping {run_dir}: {e}")
            continue

        for method in methods:
            result = compute_all_layers_volumes(hidden_states, method=method, normalize=normalize, n_axes=n_axes)
            outpath = run_dir / f"{base_no_ext}_{method}{ext}"
            torch.save(result, str(outpath))
            print(f"Saved volumes for method {method} to {outpath}")


if __name__ == '__main__':
    main()
