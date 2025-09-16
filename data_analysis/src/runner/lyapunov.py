"""Command-line wrapper for lyapunov analysis.

Reads precomputed pairwise timeseries .npz files and computes lyapunov
exponent estimates using the modular lyapunov_lib.
"""

import argparse
import os
from .lyapunov_lib import process_path, process_file, LyapunovConfig


def main():
    parser = argparse.ArgumentParser(description="Lyapunov analysis from .npz aggregated timeseries")
    parser.add_argument("input", help=".npz file or directory containing .npz files")
    parser.add_argument("--config", help="path to json config overriding defaults", default=None)
    args = parser.parse_args()

    cfg = LyapunovConfig.from_global()
    if args.config:
        import json
        with open(args.config, "r") as f:
            custom = json.load(f)
        cfg.cfg.update(custom)

    path = args.input
    if os.path.isdir(path):
        res = process_path(path, cfg)
    else:
        res = process_file(path, cfg)

    print("Lyapunov processing complete.")


if __name__ == "__main__":
    main()

