import os
import json
import numpy as np
from datetime import datetime


def make_output_dir(input_path):
    base = os.path.abspath(input_path)
    if os.path.isdir(base):
        outroot = base
    else:
        outroot = os.path.dirname(base)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir = os.path.join(outroot, "lyapunov", ts)
    os.makedirs(outdir, exist_ok=True)
    return outdir


def save_results(outdir, name, results_dict):
    outpath = os.path.join(outdir, f"{name}.json")
    def _to_json_safe(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, dict):
            return {k: _to_json_safe(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_to_json_safe(v) for v in o]
        return o

    safe = _to_json_safe(results_dict)
    with open(outpath, "w") as f:
        json.dump(safe, f, indent=2)
    return outpath


def save_npz(outdir, name, **arrays):
    outpath = os.path.join(outdir, f"{name}.npz")
    np.savez(outpath, **arrays)
    return outpath
