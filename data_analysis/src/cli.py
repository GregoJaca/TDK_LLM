# # STATUS: PARTIAL
# import argparse
# import subprocess
# from config import CONFIG

# def main():
#     parser = argparse.ArgumentParser(description="LLM Trajectory Metrics Pipeline")
#     subparsers = parser.add_subparsers(dest="command", help="Available commands")

#     # --- run_all command ---
#     parser_all = subparsers.add_parser("run_all", help="Run the full pipeline")
#     parser_all.add_argument("--input", type=str, default=CONFIG["input_path"], help="Path to the input tensor file")

#     # --- sweep command ---
#     parser_sweep = subparsers.add_parser("sweep", help="Run a hyperparameter sweep")
#     parser_sweep.add_argument("--input", type=str, default=CONFIG["input_path"], help="Path to the input tensor file")

#     # --- reduce command (placeholder) ---
#     parser_reduce = subparsers.add_parser("reduce", help="Perform dimensionality reduction")
#     parser_reduce.add_argument("--method", type=str, default=CONFIG["reduction"]["default_method"], help="Reduction method")
#     parser_reduce.add_argument("--input", type=str, required=True, help="Input tensor file")
#     parser_reduce.add_argument("--out", type=str, required=True, help="Output file for reduced tensor")

#     # --- metrics command (placeholder) ---
#     parser_metrics = subparsers.add_parser("metrics", help="Compute trajectory metrics")
#     parser_metrics.add_argument("--reduced", type=str, required=True, help="Path to the reduced tensor file")
#     # Default to empty list; metric selection is controlled by per-metric 'enabled' flags
#     parser_metrics.add_argument("--metrics", type=str, nargs='+', default=[], help="Metrics to compute (overrides config enabled flags)")

#     # --- lyapunov command (placeholder) ---
#     parser_lyapunov = subparsers.add_parser("lyapunov", help="Estimate Lyapunov exponent")
#     parser_lyapunov.add_argument("--reduced", type=str, required=True, help="Path to the reduced tensor file")

#     args = parser.parse_args()

#     # python_executable = r"C:\Users\grego\miniconda3\envs\ml_env\python.exe"
#     python_executable = r"python" # CLUSTER

#     if args.command == "run_all":
#         subprocess.run(rf"{python_executable} C:\Users\grego\OneDrive\Documents\BME_UNI_WORK\TDK_2025\code_LLM\new\chad_analysis\run_all.py --input {args.input}", shell=True)
#     elif args.command == "sweep":
#         subprocess.run(rf"{python_executable} C:\Users\grego\OneDrive\Documents\BME_UNI_WORK\TDK_2025\code_LLM\new\chad_analysis\run_sweep.py --input {args.input}", shell=True)
#     elif args.command == "reduce":
#         print(f"TODO: Implement 'reduce' command with method {args.method}")
#     elif args.command == "metrics":
#         print(f"TODO: Implement 'metrics' command for metrics {args.metrics}")
#     elif args.command == "lyapunov":
#         print(f"TODO: Implement 'lyapunov' command")
#     else:
#         parser.print_help()
