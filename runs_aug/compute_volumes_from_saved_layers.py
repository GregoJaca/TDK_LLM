import os
from pathlib import Path
import torch
from src.config import Default, Analysis
from src.analysis import compute_all_layers_volumes


def collect_hidden_states(results_dir: str, selected_layers, n_conditions):
    res = {}
    results_path = Path(results_dir)
    # auto-detect available layer indices from filenames
    found_layers = set()
    for p in results_path.glob('hidden_states_cond_*_layer_*.pt'):
        parts = p.name.split('_')
        # expected pattern: hidden_states_cond_{i}_layer_{layer_idx}.pt
        try:
            layer_pos = parts.index('layer')
            layer_idx = int(parts[layer_pos + 1].split('.pt')[0])
            found_layers.add(layer_idx)
        except Exception:
            continue

    if not found_layers:
        # fallback to provided selected_layers
        layer_candidates = list(selected_layers)
    else:
        # use detected layers (do not intersect with config; prefer actual saved files)
        layer_candidates = sorted(found_layers)

    print(f"Detected layers in {results_dir}: {sorted(found_layers)}")
    print(f"Using layer candidates: {layer_candidates}")

    for layer_idx in layer_candidates:
        layer_list = []
        missing = False
        for i in range(n_conditions):
            fname = os.path.join(results_dir, f"hidden_states_cond_{i}_layer_{layer_idx}.pt")
            if not os.path.exists(fname):
                missing = True
                break
            tensor = torch.load(fname, map_location='cpu')
            layer_list.append(tensor)
        if missing:
            # skip layers that don't have a full set of trajectories
            print(f"Skipping layer {layer_idx} in {results_dir}: missing some trajectory files")
            continue
        stacked = torch.stack(layer_list, dim=0)
        print(f"Layer {layer_idx} stacked shape: {stacked.shape}")
        res[int(layer_idx)] = stacked

    if not res:
        raise FileNotFoundError(f"No complete layers found under {results_dir}")
    return res


def main():
    base_results_dir = Path(Default.RESULTS_DIR)
    # Temporary, minimal fallback: if the configured results dir does not exist,
    # try the known run folder used during development so the script finds
    # hidden_states files without changing defaults permanently.
    if not base_results_dir.exists():
        alt = Path(r"C:/Users/grego/OneDrive/Documents/BME_UNI_WORK/TDK_2025/git_repo/TDK_LLM/runs_aug/launch_sep/interstellar_propulsion_review_0_0.00035")
        if alt.exists():
            print(f"Default results dir {base_results_dir} not found, falling back to {alt}")
            base_results_dir = alt
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
            # debug: print result shapes per layer
            for L, entry in result.items():
                vols = entry.get('volumes')
                axes = entry.get('axes')
                logv = entry.get('log_volumes')
                chosen = entry.get('chosen_n_axes')
                print(f"Layer {L} -> volumes shape: {getattr(vols,'shape',None)}, axes shape: {getattr(axes,'shape',None)}, log_vol shape: {getattr(logv,'shape',None)}, chosen_n_axes shape: {getattr(chosen,'shape',None)}")
            outpath = run_dir / f"{base_no_ext}_{method}{ext}"
            torch.save(result, str(outpath))
            print(f"Saved volumes for method {method} to {outpath}")


if __name__ == '__main__':
    main()
