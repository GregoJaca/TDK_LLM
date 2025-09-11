"""
This script performs a parameter sweep analysis for trajectory metrics.

It allows for varying a single parameter at a time and running the analysis from
'run_all_experiments.py' and/or 'compute_metric_matrices.py'.

The script is designed to be flexible and modular, allowing for easy extension
to other parameters and analyses.
"""

import os
import importlib
import argparse
from config import CONFIG
from run_all_experiments import EMBEDDING_METHODS, Experiment
import run_all
import compute_metric_matrices
from src.utils.logging import get_logger

logger = get_logger(__name__)

def run_sweep(sweep_param, sweep_values, run_experiments, compute_matrices):
    """
    Performs a parameter sweep over the given values and runs the selected analyses.
    """
    logger.info("Starting sweep analysis.")
    for rrr in Experiment.RADII:
        for TEMPERATURE in Experiment.TEMPS:
            for embedder in EMBEDDING_METHODS:
                embed_name = embedder.replace('/', '_')
                for value in sweep_values:
                    logger.info(f"Running sweep with {sweep_param}={value} for {embed_name}")

                    # Override the config for the sweep
                    if sweep_param == 'window_size':
                        CONFIG['sliding_window']['window_size'] = value
                    else:
                        # Here you can add more parameters to sweep
                        pass

                    # Expose sweep metadata for downstream modules to append to filenames
                    prev_sweep = CONFIG.get('sweep')
                    CONFIG['sweep'] = {'param': sweep_param, 'value': value}

                    results_root = os.path.normpath(
                        f"C:/Users/grego/OneDrive/Documents/BME_UNI_WORK/TDK_2025/git_repo/TDK_LLM/runs_aug/launch_aug/childhood_personality_development_{TEMPERATURE}_{rrr}/results_sweep/{embed_name}"
                    )
                    
                    input_path = os.path.normpath(
                        f"C:/Users/grego/OneDrive/Documents/BME_UNI_WORK/TDK_2025/git_repo/TDK_LLM/runs_aug/launch_aug/childhood_personality_development_{TEMPERATURE}_{rrr}/{embed_name}.pt"
                    )

                    logger.info(f"Input path: {input_path}")
                    logger.info(f"Results root: {results_root}")

                    os.makedirs(results_root, exist_ok=True)

                    if run_experiments:
                        logger.info("Calling run_all.main")
                        # Mirror run_all_experiments: temporarily set CONFIG so downstream
                        # modules that read CONFIG['results_root'] and CONFIG['input_path']
                        # (e.g. src.runner.metrics_runner) write to the intended folder.
                        prev_input = CONFIG.get("input_path")
                        prev_results = CONFIG.get("results_root")
                        CONFIG["input_path"] = input_path
                        CONFIG["results_root"] = results_root
                        try:
                            # pass sweep_param_value so run_all names plots appropriately
                            run_all.main(input_path=input_path, results_root=results_root, sweep_param_value=value)
                        except Exception as e:
                            logger.error(f"run_all.main failed: {e}")
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
                        logger.info("Finished run_all.main")

                    if compute_matrices:
                        logger.info("Calling compute_metric_matrices.main")
                        # Avoid saving .pt matrix files during sweep runs per user request
                        compute_metric_matrices.main(input_path=input_path, results_root=results_root, save_matrices=False, save_plots=True, sweep_param_value=value)
                        logger.info("Finished compute_metric_matrices.main")
                    # restore sweep metadata
                    if prev_sweep is None:
                        CONFIG.pop('sweep', None)
                    else:
                        CONFIG['sweep'] = prev_sweep
    logger.info("Sweep analysis finished.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform a parameter sweep analysis.')
    parser.add_argument('--sweep_param', type=str, default='window_size', help='The parameter to sweep.')
    parser.add_argument('--sweep_values', type=int, nargs='+', default=[1,16,32], help='The values to sweep.')
    parser.add_argument('--run_experiments', type=bool, default=True, help='Run the analysis from run_all_experiments.')
    parser.add_argument('--compute_matrices', type=bool, default=False, help='Run the analysis from compute_metric_matrices.')
    args = parser.parse_args()

    run_sweep(args.sweep_param, args.sweep_values, args.run_experiments, args.compute_matrices)