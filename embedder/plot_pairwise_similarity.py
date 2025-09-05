"""
Plot pairwise trajectory cosine similarity over time-step.

Usage (from the embedder folder):
    activate ml_env
    python plot_pairwise_similarity.py path/to/model_embeddings.pt --meta path/to/meta.json

If no --meta is provided, the script assumes all trajectories have full length = tensor.shape[1]

Generates two files next to the .pt file:
  <prefix>_pairwise_linear.png
  <prefix>_pairwise_logy.png

Notes:
- Cosine similarity is computed per time-step between two trajectories' embedding vectors.
- For the log plot the values are shifted by +1 to make them positive (cos in [-1,1] -> [0,2])
  then a tiny eps is added to avoid log(0).
"""
from __future__ import annotations

import argparse
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import matplotlib.pyplot as plt


def l2_normalize(a: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(a, axis=-1, keepdims=True)
    norms = np.maximum(norms, eps)
    return a / norms


def load_tensor(pt_path: Path) -> np.ndarray:
    t = torch.load(str(pt_path), map_location="cpu")
    if not isinstance(t, torch.Tensor):
        raise ValueError("Loaded object is not a torch.Tensor")
    return t.cpu().numpy()


def compute_pairwise_timewise_cos(tensor: np.ndarray, lengths: Optional[list] = None, eps: float = 1e-12):
    # tensor: [n_traj, len_traj, dim]
    n, L, d = tensor.shape
    # ensure float32
    arr = tensor.astype(np.float32)

    # pre-normalize along dim for stable dot product (but keep zeros as-is)
    norms = np.linalg.norm(arr, axis=-1)
    safe_norms = np.maximum(norms, eps)
    arr_normed = arr / safe_norms[..., None]

    pairs = list(combinations(range(n), 2))
    sims_by_pair = {}

    for i, j in pairs:
        sims = np.full((L,), np.nan, dtype=np.float32)
        for t in range(L):
            if lengths is not None:
                if t >= lengths[i] or t >= lengths[j]:
                    sims[t] = np.nan
                    continue
            v1 = arr_normed[i, t]
            v2 = arr_normed[j, t]
            # if either vector is effectively zero (original norm < eps) set nan
            if norms[i, t] < eps or norms[j, t] < eps:
                sims[t] = np.nan
                continue
            sims[t] = float(np.dot(v1, v2))
        sims_by_pair[(i, j)] = sims

    return sims_by_pair


def plot_pairs(sims_by_pair: dict, out_prefix: Path, max_len: int, max_pairs: Optional[int] = None):
    x = np.arange(max_len)
    pairs = list(sims_by_pair.keys())
    if max_pairs is not None and len(pairs) > max_pairs:
        pairs = pairs[:max_pairs]

    # Linear plot
    plt.figure(figsize=(10, 6))
    for (i, j) in pairs:
        sims = sims_by_pair[(i, j)]
        plt.plot(x, sims, label=f"{i}-{j}", linewidth=1)
    plt.xlabel("time-step (trajectory position)")
    plt.ylabel("cosine similarity")
    plt.title("Pairwise cosine similarity over time (linear)")
    if len(pairs) <= 30:
        plt.legend(fontsize="small", ncol=2)
    plt.grid(alpha=0.3)
    out_linear = out_prefix.with_name(out_prefix.name + "_pairwise_linear.png")
    plt.tight_layout()
    plt.savefig(out_linear, dpi=150)
    plt.close()

    # Log-y plot: shift +1 to make values positive in [0,2], add small eps to avoid log(0)
    eps = 1e-9
    plt.figure(figsize=(10, 6))
    for (i, j) in pairs:
        sims = sims_by_pair[(i, j)]
        sims_shift = sims + 1.0
        # keep NaNs as-is; matplotlib will skip them
        plt.plot(x, sims_shift + eps, label=f"{i}-{j}", linewidth=1)
    plt.xlabel("time-step (trajectory position)")
    plt.ylabel("cosine similarity + 1 (log scale)")
    plt.title("Pairwise cosine similarity over time (y-axis log)")
    plt.yscale("log")
    if len(pairs) <= 30:
        plt.legend(fontsize="small", ncol=2)
    plt.grid(alpha=0.3)
    out_log = out_prefix.with_name(out_prefix.name + "_pairwise_logy.png")
    plt.tight_layout()
    plt.savefig(out_log, dpi=150)
    plt.close()

    return out_linear, out_log


def main():
    p = argparse.ArgumentParser(description="Plot pairwise trajectory cosine similarity over time")
    p.add_argument("pt", nargs="?", help="Path to the .pt tensor file saved by the embedding pipeline. If omitted or a directory is provided, all .pt files in the directory will be processed.")
    p.add_argument("--meta", help="Optional metadata json (containing 'lengths' list)")
    p.add_argument("--max-pairs", type=int, default=None, help="Plot only the first N pairs (for many trajectories)")
    # By default create a different plot per pair (user requested this behavior).
    p.add_argument("--per-pair", dest="per_pair", action="store_true", help="Save a separate plot for each pair of trajectories (two files per pair: linear and log). If false, all pairs are overlaid in two plots as before.")
    p.add_argument("--no-per-pair", dest="per_pair", action="store_false", help="Create combined plots that overlay all pairs (opposite of --per-pair)")
    p.set_defaults(per_pair=True)
    p.add_argument("--outputs-dir", default="outputs", help="Directory to scan for .pt files when --pt is omitted or a directory is provided")
    args = p.parse_args()

    # Determine list of .pt files to process
    pt_inputs: list[Path] = []
    if args.pt:
        pt_path = Path(args.pt)
        if pt_path.is_dir():
            pt_inputs = sorted(pt_path.glob("*.pt"))
        else:
            if not pt_path.exists():
                raise SystemExit(f"File not found: {pt_path}")
            pt_inputs = [pt_path]
    else:
        out_dir = Path(args.outputs_dir)
        if not out_dir.exists():
            raise SystemExit(f"Outputs directory not found: {out_dir}")
        pt_inputs = sorted(out_dir.glob("*.pt"))

    if len(pt_inputs) == 0:
        raise SystemExit("No .pt files found to process.")

    for pt_path in pt_inputs:
        print(f"Processing: {pt_path}")
        tensor = load_tensor(pt_path)
        n, L, d = tensor.shape

        lengths = None
        # try to find a matching meta file next to the .pt
        meta_candidate = pt_path.with_name(pt_path.stem + "_meta.json")
        if args.meta:
            meta_candidate = Path(args.meta)
        if meta_candidate.exists():
            try:
                with open(meta_candidate, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                if "lengths" in meta and isinstance(meta["lengths"], list):
                    lengths = meta["lengths"]
            except Exception:
                print("Warning: failed to read meta file; proceeding without lengths mask.")

        sims_by_pair = compute_pairwise_timewise_cos(tensor, lengths=lengths)

        out_prefix = pt_path.with_suffix("")
        if args.per_pair:
            # create individual plots per pair
            for (i, j), sims in sims_by_pair.items():
                pair_dict = {(i, j): sims}
                pair_prefix = out_prefix.with_name(f"{out_prefix.name}_pair_{i}-{j}")
                out_linear, out_log = plot_pairs(pair_dict, pair_prefix, max_len=L, max_pairs=1)
                print(f"Saved linear plot -> {out_linear}")
                print(f"Saved log-y plot -> {out_log}")
        else:
            out_linear, out_log = plot_pairs(sims_by_pair, out_prefix, max_len=L, max_pairs=args.max_pairs)
            print(f"Saved linear plot -> {out_linear}")
            print(f"Saved log-y plot -> {out_log}")


if __name__ == "__main__":
    main()
