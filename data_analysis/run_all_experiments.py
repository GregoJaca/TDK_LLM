

class Experiment:
    # RADII = [0.0003, 0.0004]
    # TEMPS = [0, 0.6]
    RADII = [0.0004]
    TEMPS = [0]
EMBEDDING_METHODS = [
    # "sentence-transformers/all-mpnet-base-v2",
    # "intfloat/e5-large-v2",
    # "facebook/contriever",
    "hidden_states_layer_-1",
    # "hidden_states_layer_14",
    # "hidden_states_layer_22"
]
import os
from config import CONFIG

import run_all


def run_all_experiments():
    """Iterate over experiments, override CONFIG per-run and invoke run_all.main.

    This reuses the implementation in run_all.py by temporarily setting
    CONFIG['input_path'] and CONFIG['results_root'] for each experiment.
    """
    for rrr in Experiment.RADII:
        for TEMPERATURE in Experiment.TEMPS:
            for embedder in EMBEDDING_METHODS:
                RADIUS_INITIAL_CONDITIONS = rrr
                embed_name = embedder.replace('/', '_')
                input_path = os.path.normpath(
                    f"C:/Users/grego/OneDrive/Documents/BME_UNI_WORK/TDK_2025/git_repo/TDK_LLM/runs_aug/launch_aug/childhood_personality_development_{TEMPERATURE}_{RADIUS_INITIAL_CONDITIONS}/{embed_name}.pt"
                )
                results_root = os.path.normpath(
                    f"C:/Users/grego/OneDrive/Documents/BME_UNI_WORK/TDK_2025/git_repo/TDK_LLM/runs_aug/launch_aug/childhood_personality_development_{TEMPERATURE}_{RADIUS_INITIAL_CONDITIONS}/results/{embed_name}"
                )

                # Ensure results directory exists
                os.makedirs(results_root, exist_ok=True)

                # Temporarily override CONFIG values used by run_all.main
                prev_input = CONFIG.get("input_path")
                prev_results = CONFIG.get("results_root")
                CONFIG["input_path"] = input_path
                CONFIG["results_root"] = results_root

                print(f"Running experiment: temp={TEMPERATURE}, radius={RADIUS_INITIAL_CONDITIONS}, embedder={embedder}")
                try:
                    # pass input_path explicitly (run_all will still use CONFIG for results_root)
                    run_all.main(input_path=input_path)
                except Exception as e:
                    # Keep this minimal: report and continue with next experiment
                    print(f"Experiment failed for {input_path}: {e}")
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


if __name__ == "__main__":
    run_all_experiments()