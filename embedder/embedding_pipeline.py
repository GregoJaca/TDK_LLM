"""
Main pipeline entrypoint.

Reads the list of strings from JSON (top-level array of strings),
runs each model extractor, pads/truncates sequences per-method and saves tensors + metadata.
"""

import os
import json
import time
import argparse
from typing import List, Any, Dict

import numpy as np
import torch

import config as cfg
from utils import save_tensor_and_meta, save_single_trajectory, slugify_model_id, pad_sequences_to_tensor
from models import get_extractor

from src.config import Default, ModelConfig, Prompts, Experiment


# Helper to determine torch dtype
def get_torch_dtype(dtype_str: str):
    if dtype_str == "float16":
        return torch.float16
    return torch.float32

def load_input_texts(path: str) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input JSON not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a top-level array of strings.")
    # ensure each element is a string
    texts = [str(x) for x in data]
    return texts

def process_model(model_id: str, texts: List[str], cfg_module) -> None:
    print(f"=== Processing model: {model_id} ===")
    extractor = get_extractor(model_id)
    try:
        arrays = extractor(texts, cfg_module)
    except Exception as e:
        msg = f"Failed to initialize or run extractor for model {model_id}: {e}"
        if cfg_module.ON_LOAD_FAIL == "skip":
            print(msg)
            print(f"Skipping model {model_id}.")
            return
        else:
            raise

    # arrays: list of np.ndarray per trajectory [len_i, dim]
    if len(arrays) == 0:
        print(f"No output from extractor for model {model_id}. Skipping.")
        return

    # verify embedding dims are consistent
    dims = [arr.shape[1] if arr.size != 0 else None for arr in arrays]
    # If some are zero-length, infer dimension from first non-empty
    emb_dim = None
    for d in dims:
        if d is not None:
            emb_dim = d
            break
    if emb_dim is None:
        print(f"All trajectories empty for model {model_id}. Saving empty tensor.")
        # create empty tensor
        tensor = torch.empty((len(arrays), 0, 0), dtype=get_torch_dtype(cfg_module.DTYPE))
        meta = {
            "model_id": model_id,
            "n_traj": len(arrays),
            "lengths": [0] * len(arrays),
            "emb_dim": 0,
            "note": "All trajectories empty"
        }
        save_tensor_and_meta(tensor, meta, cfg_module.OUTPUT_DIR, model_id, cfg_module)
        return
    # ensure all embeddings have same dim
    for i, arr in enumerate(arrays):
        if arr.size == 0:
            continue
        if arr.shape[1] != emb_dim:
            raise ValueError(f"Inconsistent embedding dimension for model {model_id} at traj {i}: {arr.shape[1]} vs {emb_dim}")

    # optional normalization (numpy)
    if cfg_module.L2_NORMALIZE:
        arrays = [ (arr if arr.size==0 else np.asarray(arr, dtype=np.float32) / (np.linalg.norm(arr, axis=1, keepdims=True)+1e-12) ) for arr in arrays ]

    # If configured, save each trajectory separately as it's computed to avoid keeping everything in memory
    save_per_traj = getattr(cfg_module, "SAVE_PER_TRAJ", True)
    if save_per_traj:
        print(f"Saving {len(arrays)} trajectories individually for model {model_id}...")
        # Build a brief per-model meta summary
        model_meta = {
            "model_id": model_id,
            "n_traj": len(arrays),
            "emb_dim": emb_dim,
            "padding_policy": cfg_module.PADDING_POLICY,
            "config_snapshot": {
                "EMBEDDING_METHODS": cfg_module.EMBEDDING_METHODS,
                "DEFAULT_TIME_MODE": cfg_module.DEFAULT_TIME_MODE,
                "WINDOW_SIZE_TOKENS": cfg_module.WINDOW_SIZE_TOKENS,
                "WINDOW_STRIDE_TOKENS": cfg_module.WINDOW_STRIDE_TOKENS,
                "PREFIX_STEP_TOKENS": cfg_module.PREFIX_STEP_TOKENS,
                "L2_NORMALIZE": cfg_module.L2_NORMALIZE,
                "DTYPE": cfg_module.DTYPE
            },
            "timestamp": time.time()
        }

        # Save aggregated model meta as well
        meta_path = os.path.join(cfg_module.OUTPUT_DIR, cfg_module.META_FILENAME_TEMPLATE.format(model_slug=slugify_model_id(model_id)))
        os.makedirs(cfg_module.OUTPUT_DIR, exist_ok=True)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(model_meta, f, indent=2)

        # Save each trajectory
        for i, arr in enumerate(arrays):
            traj_meta = {
                "model_id": model_id,
                "traj_index": i,
                "length": int(arr.shape[0]) if getattr(arr, 'shape', None) else 0,
                "emb_dim": int(arr.shape[1]) if getattr(arr, 'shape', None) and len(arr.shape) > 1 else emb_dim,
                "timestamp": time.time()
            }
            save_single_trajectory(arr, traj_meta, cfg_module.OUTPUT_DIR, model_id, cfg_module, traj_idx=i)
        print(f"Saved {len(arrays)} per-trajectory files for {model_id} in {cfg_module.OUTPUT_DIR}")
        return

    # pad/truncate to tensor (fallback preserved behavior)
    torch_dtype = get_torch_dtype(cfg_module.DTYPE)
    tensor, lengths = pad_sequences_to_tensor(arrays, dtype=torch_dtype, padding_policy=cfg_module.PADDING_POLICY, max_length_truncate=cfg_module.MAX_LENGTH_TRUNCATE)
    # Save
    meta = {
        "model_id": model_id,
        "n_traj": len(arrays),
        "lengths": lengths,
        "emb_dim": emb_dim,
        "padding_policy": cfg_module.PADDING_POLICY,
        "padding_info": {
            "max_length": tensor.shape[1]
        },
        "config_snapshot": {
            "EMBEDDING_METHODS": cfg_module.EMBEDDING_METHODS,
            "DEFAULT_TIME_MODE": cfg_module.DEFAULT_TIME_MODE,
            "WINDOW_SIZE_TOKENS": cfg_module.WINDOW_SIZE_TOKENS,
            "WINDOW_STRIDE_TOKENS": cfg_module.WINDOW_STRIDE_TOKENS,
            "PREFIX_STEP_TOKENS": cfg_module.PREFIX_STEP_TOKENS,
            "L2_NORMALIZE": cfg_module.L2_NORMALIZE,
            "DTYPE": cfg_module.DTYPE
        },
        "timestamp": time.time()
    }

    save_tensor_and_meta(tensor, meta, cfg_module.OUTPUT_DIR, model_id, cfg_module)
    print(f"Saved embeddings for {model_id} -> {slugify_model_id(model_id)}.pt  (shape {tuple(tensor.shape)})")

def main():


    for rrr in Experiment.RADII:
        for ttt in range(len(Experiment.TEMPS)):
            for prompt_idx in range(len(Prompts.prompts)):
                config = Default()
                config.TEMPERATURE = Experiment.TEMPS[ttt]
                config.RESULTS_DIR = f"../{Default.RESULTS_DIR}/{Prompts.prompt_names[prompt_idx]}_{config.TEMPERATURE}_{rrr}/"

                cfg.INPUT_JSON_PATH = f"{config.RESULTS_DIR}/generated_text.json"
                cfg.OUTPUT_DIR = f"{config.RESULTS_DIR}"

                texts = load_input_texts(cfg.INPUT_JSON_PATH)
                print(f"Loaded {len(texts)} trajectories from {cfg.INPUT_JSON_PATH}")

                # iterate models
                for model_id in cfg.EMBEDDING_METHODS:
                    try:
                        process_model(model_id, texts, cfg)
                    except Exception as e:
                        print(f"Error processing model {model_id}: {e}")
                        if cfg.ON_LOAD_FAIL == "raise":
                            raise
                        else:
                            print(f"Skipping {model_id} due to error.")

                print("All requested models processed (or skipped on error). Outputs in:", cfg.OUTPUT_DIR)

if __name__ == "__main__":
    main()
