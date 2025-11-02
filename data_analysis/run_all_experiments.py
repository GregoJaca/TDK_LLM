

class Experiment:
    # RADII = [0.0003, 0.0004]
    # TEMPS = [0, 0.6]
    RADII = [0.00035]
    TEMPS = [0]
EMBEDDING_METHODS = [
    # kept for backward-compat in-file but will be read from config.EMBEDDING_CONFIG
]
import os
from config import CONFIG

EMBEDDING_CONFIG = CONFIG.get('EMBEDDING_CONFIG', {})
EMBEDDING_METHODS = EMBEDDING_CONFIG.get('embedding_methods', [])
INPUT_MODE = EMBEDDING_CONFIG.get('input_mode', 'single_file')
INPUT_TEMPLATE = EMBEDDING_CONFIG.get('input_template', '{embed}.pt')
import run_all
print(f"[run_all_experiments module] __name__={__name__} cwd={os.getcwd()}")


def run_all_experiments():
    """Iterate over experiments, override CONFIG per-run and invoke run_all.main.

    This reuses the implementation in run_all.py by temporarily setting
    CONFIG['input_path'] and CONFIG['results_root'] for each experiment.
    """
    embed_methods = EMBEDDING_CONFIG.get('embedding_methods', [])
    print("[run_all_experiments] module loaded. EMBEDDING_METHODS=", embed_methods)
    print("[run_all_experiments] INPUT_MODE=", INPUT_MODE, "INPUT_TEMPLATE=", INPUT_TEMPLATE)
    print("[run_all_experiments] Experiment.RADII=", Experiment.RADII)
    print("[run_all_experiments] Experiment.TEMPS=", Experiment.TEMPS)

    # If no embedding_methods provided, try to derive a single embedder from CONFIG['input_path']
    if not embed_methods:
        fallback_input = CONFIG.get('input_path')
        if isinstance(fallback_input, str) and os.path.exists(fallback_input):
            embed_guess = os.path.splitext(os.path.basename(fallback_input))[0]
            print(f"[run_all_experiments] EMBEDDING_METHODS empty; falling back to embedder={embed_guess}")
            embed_methods = [embed_guess]
        else:
            print("[run_all_experiments] EMBEDDING_METHODS empty and no valid CONFIG['input_path'] to infer embedder. Nothing to run.")
            return

    # wrap overall loop so unexpected exceptions are visible per-embedding
    for rrr in Experiment.RADII:
        for TEMPERATURE in Experiment.TEMPS:
            for embedder in embed_methods:
                RADIUS_INITIAL_CONDITIONS = rrr
                embed_name = embedder.replace('/', '_')
                run_folder = f"C:/Users/grego/OneDrive/Documents/BME_UNI_WORK/TDK_2025/git_repo/TDK_LLM/launch_sep/interstellar_propulsion_review_{TEMPERATURE}_{RADIUS_INITIAL_CONDITIONS}"

                # run_folder = f"/home/grego/LLM/launch_sep/interstellar_propulsion_review_{TEMPERATURE}_{rrr}" # CLUSTER


                if INPUT_MODE == 'per_trajectory':
                    # pass the run folder; run_all will build per-trajectory filenames using
                    # CONFIG['current_embed_raw'] which we set below
                    input_path = os.path.normpath(run_folder)
                else:
                    # legacy single-file mode
                    input_path = os.path.normpath(os.path.join(run_folder, f"{embed_name}.pt"))

                # Place results under a top-level 'results' folder so structure is:
                # .../launch_aug/results/<case_name>/<embed_name>/
                results_root = os.path.normpath(os.path.join(run_folder, "results", embed_name))

                # Ensure results directory exists
                os.makedirs(results_root, exist_ok=True)

                # Temporarily override CONFIG values used by run_all.main
                prev_input = CONFIG.get("input_path")
                prev_results = CONFIG.get("results_root")
                CONFIG["input_path"] = input_path
                CONFIG["results_root"] = results_root

                print(f"Running experiment: temp={TEMPERATURE}, radius={RADIUS_INITIAL_CONDITIONS}, embedder={embedder}")

                try:
                    # For per-trajectory mode, set a temporary CONFIG key with the raw embed template
                    prev_embed_raw = CONFIG.get('current_embed_raw')
                    if INPUT_MODE == 'per_trajectory':
                        CONFIG['current_embed_raw'] = embedder

                    # Use CONFIG['pairwise']['save_pairwise_aggregated'] directly

                    # pass input_path and results_root explicitly
                    print(f"[run_all_experiments] calling run_all.main input_path={input_path} results_root={results_root}")
                    run_all.main(input_path=input_path, results_root=results_root)

                except Exception as e:
                    # Print full traceback to help debugging
                    import traceback
                    print(f"Experiment failed for {input_path}: {e}")
                    traceback.print_exc()

                finally:
                    # restore CONFIG
                    if prev_input is None:
                        CONFIG.pop("input_path", None)
                    else:
                        CONFIG["input_path"] = prev_input
                    if prev_results is None:
                        CONFIG.pop("results_root", None)
                    else:
                        CONFIG["results_root"] = prev_results
                    # no local pairwise state to restore
                    # restore embed raw
                    if INPUT_MODE == 'per_trajectory':
                        if prev_embed_raw is None:
                            CONFIG.pop('current_embed_raw', None)
                        else:
                            CONFIG['current_embed_raw'] = prev_embed_raw


if __name__ == "__main__":
    run_all_experiments()