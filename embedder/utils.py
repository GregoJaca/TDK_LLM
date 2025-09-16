import os
import json
import math
from typing import List, Tuple, Dict, Any
import torch
import numpy as np
import nltk

# ensure punkt is available for sentence splitting
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

def slugify_model_id(model_id: str) -> str:
    return model_id.replace("/", "_").replace(":", "_")

def save_tensor_and_meta(tensor: torch.Tensor, meta: Dict[str, Any], out_dir: str, model_id: str, cfg):
    os.makedirs(out_dir, exist_ok=True)
    slug = slugify_model_id(model_id)
    pt_path = os.path.join(out_dir, cfg.OUTPUT_FILENAME_TEMPLATE.format(model_slug=slug))
    meta_path = os.path.join(out_dir, cfg.META_FILENAME_TEMPLATE.format(model_slug=slug))

    torch.save(tensor, pt_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def save_single_trajectory(tensor_or_array, meta: Dict[str, Any], out_dir: str, model_id: str, cfg, traj_idx: int):
    """
    Save a single trajectory as its own .pt file and a per-trajectory meta JSON.
    `tensor_or_array` can be a torch.Tensor or numpy.ndarray of shape [L, D].
    Files will be named '{model_slug}_traj{traj_idx}.pt' and '{model_slug}_traj{traj_idx}_meta.json'.
    """
    os.makedirs(out_dir, exist_ok=True)
    slug = slugify_model_id(model_id)
    base_pt_name = cfg.PER_TRAJ_FILENAME_TEMPLATE if hasattr(cfg, 'PER_TRAJ_FILENAME_TEMPLATE') else "{model_slug}_traj{traj_idx}.pt"
    base_meta_name = cfg.PER_TRAJ_META_TEMPLATE if hasattr(cfg, 'PER_TRAJ_META_TEMPLATE') else "{model_slug}_traj{traj_idx}_meta.json"

    pt_path = os.path.join(out_dir, base_pt_name.format(model_slug=slug, traj_idx=traj_idx))
    meta_path = os.path.join(out_dir, base_meta_name.format(model_slug=slug, traj_idx=traj_idx))

    # Convert numpy arrays to torch tensors
    if isinstance(tensor_or_array, np.ndarray):
        tensor_to_save = torch.from_numpy(tensor_or_array)
    else:
        tensor_to_save = tensor_or_array

    torch.save(tensor_to_save, pt_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

def mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    hidden_states: [batch, seq_len, dim]
    attention_mask: [batch, seq_len] (0/1)
    returns: [batch, dim] mean-pooled across non-masked tokens
    """
    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)  # [batch, seq_len, 1]
    summed = (hidden_states * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts

def l2_normalize_np(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=axis, keepdims=True)
    norms = np.maximum(norms, eps)
    return x / norms

def pad_sequences_to_tensor(list_of_arrays: List[np.ndarray], dtype: torch.dtype, padding_policy: str, max_length_truncate: int = None) -> Tuple[torch.Tensor, List[int]]:
    """
    list_of_arrays: each element is np.ndarray shape [len_i, dim]
    Returns: tensor [n, max_len, dim] padded with zeros or truncated, and lengths list
    """
    n = len(list_of_arrays)
    if n == 0:
        return torch.empty((0, 0, 0), dtype=dtype), []

    lengths = [arr.shape[0] for arr in list_of_arrays]
    dim = list_of_arrays[0].shape[1]
    max_len = max(lengths)

    if padding_policy == "truncate" and max_length_truncate is not None:
        max_len = min(max_len, max_length_truncate)

    out = np.zeros((n, max_len, dim), dtype=np.float32)
    for i, arr in enumerate(list_of_arrays):
        L = arr.shape[0]
        if padding_policy == "truncate" and max_length_truncate is not None and L > max_length_truncate:
            out[i, :max_length_truncate, :] = arr[:max_length_truncate]
            lengths[i] = max_length_truncate
        else:
            out[i, :L, :] = arr

    tensor = torch.from_numpy(out).to(dtype=dtype)
    return tensor, lengths

def text_to_token_windows(tokenizer, text: str, window_size: int, stride: int, add_special_tokens: bool = True) -> List[List[int]]:
    """
    Returns list of token-id windows (each a list of ints) for the provided tokenizer.
    This is purely token-id based; decoding is done elsewhere if needed.
    """
    enc = tokenizer.encode(text, add_special_tokens=add_special_tokens)
    n = len(enc)
    if n == 0:
        return []
    windows = []
    if n <= window_size:
        windows.append(enc)
        return windows
    i = 0
    while i < n:
        start = i
        end = min(i + window_size, n)
        windows.append(enc[start:end])
        if end == n:
            break
        i += stride
    return windows

def decode_token_ids_to_text(tokenizer, ids: List[int]) -> str:
    # avoid special tokens if present
    try:
        txt = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    except Exception:
        # fallback: join tokens as strings
        txt = " ".join(str(i) for i in ids)
    return txt

def flatten(list_of_lists):
    return [el for sub in list_of_lists for el in sub]
