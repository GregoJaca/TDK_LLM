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
    results_dir = Default.RESULTS_DIR
    selected_layers = Default.SELECTED_LAYERS
    n_conditions = Default.N_INITIAL_CONDITIONS
    if not os.path.isdir(results_dir):
        raise NotADirectoryError(f"Results dir not found: {results_dir}")

    hidden_states = collect_hidden_states(results_dir, selected_layers, n_conditions)

    methods = Analysis.LAYER_VOLUME_METHOD
    normalize = Analysis.NORMALIZE_HIDDEN_STATES
    n_axes = Analysis.LAYER_VOLUME_N_AXES

    os.makedirs(results_dir, exist_ok=True)
    base = Analysis.LAYER_VOLUME_OUTPUT_FILENAME
    base_no_ext, ext = os.path.splitext(base)
    if ext == '':
        ext = '.pt'

    for method in methods:
        result = compute_all_layers_volumes(hidden_states, method=method, normalize=normalize, n_axes=n_axes)
        outpath = os.path.join(results_dir, f"{base_no_ext}_{method}{ext}")
        torch.save(result, outpath)
        print(f"Saved volumes for method {method} to {outpath}")


if __name__ == '__main__':
    main()
