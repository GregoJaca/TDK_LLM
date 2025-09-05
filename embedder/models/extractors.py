"""
Per-model extractor utilities.

Each extractor implements a function:
    embed_texts(texts: List[str], cfg) -> np.ndarray list of arrays per-trajectory
The pipeline expects for each model a list (len = n_traj) of np.ndarray of shape [len_traj_i, emb_dim].

We provide a get_extractor(model_id) factory that returns a callable extractor(texts, cfg).
"""

import os
import math
import json
import numpy as np
import torch
from typing import List, Callable, Dict, Any, Tuple

from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
from transformers.modeling_outputs import BaseModelOutput
from utils import mean_pool, decode_token_ids_to_text, text_to_token_windows

# attempt to reuse torch device setting at runtime
def get_device(cfg):
    if cfg.DEVICE == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg.DEVICE)

def _try_model_load(model_id: str, cfg) -> Tuple[Any, Any]:
    """
    Tries to load model and tokenizer. Returns (model, tokenizer).
    For sentence-transformers models we return (SentenceTransformer instance, None).
    """
    device = get_device(cfg)
    hf_kwargs = {}
    if cfg.HF_TOKEN:
        hf_kwargs["use_auth_token"] = cfg.HF_TOKEN
    # trust_remote_code option
    trust = cfg.TRUST_REMOTE_CODE

    # sentence-transformers special case
    if model_id.startswith("sentence-transformers/"):
        try:
            m = SentenceTransformer(model_id, device=str(device))
            return m, None
        except Exception as e:
            raise RuntimeError(f"Failed to load sentence-transformer {model_id}: {e}")

    # Otherwise try transformers AutoModel + AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, **hf_kwargs)
        model = AutoModel.from_pretrained(model_id, trust_remote_code=trust, **hf_kwargs)
        model.to(device)
        model.eval()
        return model, tokenizer
    except Exception as e:
        # try again with trust_remote_code True if not tried
        if not trust:
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, **hf_kwargs)
                model = AutoModel.from_pretrained(model_id, trust_remote_code=True, **hf_kwargs)
                model.to(device)
                model.eval()
                return model, tokenizer
            except Exception as e2:
                raise RuntimeError(f"Failed to load model {model_id}: {e2}")
        raise RuntimeError(f"Failed to load model {model_id}: {e}")

def _mean_pool_numpy_from_torch(hidden, mask):
    """
    hidden: torch.Tensor [batch, seq, dim]
    mask: torch.Tensor [batch, seq]
    returns np.ndarray [batch, dim]
    """
    pooled = mean_pool(hidden, mask)
    return pooled.detach().cpu().numpy()

def _embed_with_transformers_token_windows(model, tokenizer, texts: List[str], cfg, mode="sliding_windows"):
    """
    Generic token-windowed embedding for a transformers encoder model.
    Returns list of np.ndarray per trajectory: [len_traj_i, emb_dim]
    mode: 'sliding_windows' or 'prefix' or 'token'
    """
    device = get_device(cfg)
    batch_size = cfg.BATCH_SIZE if cfg.BATCH_SIZE else 16
    results = []
    for text in texts:
        if text is None:
            results.append(np.zeros((0, model.config.hidden_size), dtype=np.float32))
            continue
        if mode == "sentence":
            # fallback: use tokenizer to split into sentences via decode windows per sentence is complicated
            # we will use a single chunk (whole text)
            encoded = tokenizer(text, return_tensors="pt", truncation=cfg.TOKENIZER_TRUNCATION)
            input_ids = encoded["input_ids"][0].tolist()
            windows = [input_ids]
        elif mode == "token":
            # token-level: use whole token sequence and produce per-token vectors
            encoded = tokenizer(text, return_tensors="pt", truncation=False, add_special_tokens=True)
            input_ids = encoded["input_ids"][0].tolist()
            windows = [input_ids]  # mark whole sequence and then extract token-level outputs
        elif mode == "prefix":
            enc = tokenizer.encode(text, add_special_tokens=True)
            windows = []
            step = cfg.PREFIX_STEP_TOKENS
            lengths = list(range(step, len(enc) + 1, step))
            for L in lengths:
                windows.append(enc[:L])
        else:  # sliding_windows
            enc = tokenizer.encode(text, add_special_tokens=True)
            windows = text_to_token_windows(tokenizer, text, cfg.WINDOW_SIZE_TOKENS, cfg.WINDOW_STRIDE_TOKENS, add_special_tokens=True)

        # if token-mode and whole sequence, handle differently: forward entire sequence and take last_hidden_state per token
        if mode == "token":
            if len(windows) == 0 or len(windows[0]) == 0:
                results.append(np.zeros((0, model.config.hidden_size), dtype=np.float32))
                continue
            # forward in batches if long
            ids = windows[0]
            all_token_vectors = []
            i = 0
            max_len = len(ids)
            while i < max_len:
                chunk_ids = ids[i: i + cfg.WINDOW_SIZE_TOKENS]
                batch = tokenizer.pad({"input_ids": [chunk_ids]}, return_tensors="pt", padding=cfg.TOKENIZER_PADDING, truncation=cfg.TOKENIZER_TRUNCATION).to(device)
                with torch.no_grad():
                    out = model(**batch, output_hidden_states=False, return_dict=True)
                    last_hidden = out.last_hidden_state  # [1, seq, dim]
                # append token vectors
                arr = last_hidden[0].detach().cpu().numpy()
                all_token_vectors.append(arr)
                i += cfg.WINDOW_SIZE_TOKENS
            if len(all_token_vectors) == 0:
                results.append(np.zeros((0, model.config.hidden_size), dtype=np.float32))
            else:
                arr = np.vstack(all_token_vectors)
                results.append(arr)
            continue

        # For windows list: each element is a list of token ids. We'll create batches of windows (decoded as text or pass token ids)
        embeddings_per_windows = []
        # We'll forward by batching windows as token_ids
        idx = 0
        while idx < len(windows):
            batch_windows = windows[idx: idx + batch_size]
            # Prepare input batch
            # each window is a list of ids; use tokenizer.pad on dict of input_ids
            batch_inputs = tokenizer.pad({"input_ids": batch_windows}, return_tensors="pt", padding=cfg.TOKENIZER_PADDING, truncation=cfg.TOKENIZER_TRUNCATION)
            batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
            with torch.no_grad():
                out = model(**batch_inputs, output_hidden_states=False, return_dict=True)
                # try model-provided pooling if exists
                pooled = None
                # Many models don't have dedicated embedding head; use mean pooling over last_hidden_state
                if hasattr(model, "get_sentence_embedding") and callable(getattr(model, "get_sentence_embedding")):
                    # some models like bge may expose this
                    try:
                        pooled_t = model.get_sentence_embedding(**batch_inputs)
                        if isinstance(pooled_t, torch.Tensor):
                            pooled = pooled_t.detach().cpu().numpy()
                        else:
                            pooled = np.asarray(pooled_t)
                    except Exception:
                        pooled = None
                if pooled is None:
                    # fallback mean-pool last hidden
                    last_hidden = out.last_hidden_state  # [batch, seq, dim]
                    mask = batch_inputs.get("attention_mask", torch.ones(last_hidden.shape[:2], device=last_hidden.device))
                    pooled_t = mean_pool(last_hidden, mask)
                    pooled = pooled_t.detach().cpu().numpy()
            embeddings_per_windows.extend(pooled)
            idx += batch_size

        arr = np.asarray(embeddings_per_windows, dtype=np.float32)
        results.append(arr)

    return results

def _embed_with_sentence_transformer(model_id: str, texts: List[str], cfg):
    """
    Uses SentenceTransformer.encode to embed either sentences or (decoded) windows.
    Returns list of np.ndarray per trajectory
    """
    device = get_device(cfg)
    # load model
    model = SentenceTransformer(model_id, device=str(device))
    batch_size = cfg.BATCH_SIZE

    results = []
    for text in texts:
        if cfg.DEFAULT_TIME_MODE == "sentence" and cfg.SENTENCE_SPLIT:
            from nltk.tokenize import sent_tokenize
            sents = sent_tokenize(text)
            if len(sents) == 0:
                results.append(np.zeros((0, model.get_sentence_embedding_dimension()), dtype=np.float32))
                continue
            emb = model.encode(sents, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False)
            results.append(emb.astype(np.float32))
        elif cfg.DEFAULT_TIME_MODE == "prefix":
            # cumulative prefixes at token-level via tokenizer decode
            tokenizer = model.tokenizer
            enc = tokenizer.encode(text, add_special_tokens=True)
            step = cfg.PREFIX_STEP_TOKENS
            prefixes = []
            Ls = list(range(step, len(enc) + 1, step))
            for L in Ls:
                prefixes.append(tokenizer.decode(enc[:L], skip_special_tokens=True))
            if len(prefixes) == 0:
                results.append(np.zeros((0, model.get_sentence_embedding_dimension()), dtype=np.float32))
                continue
            emb = model.encode(prefixes, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False)
            results.append(emb.astype(np.float32))
        else:
            # sliding_windows on token-level: create decoded token windows then encode
            tokenizer = model.tokenizer
            windows_ids = text_to_token_windows(tokenizer, text, cfg.WINDOW_SIZE_TOKENS, cfg.WINDOW_STRIDE_TOKENS, add_special_tokens=True)
            if len(windows_ids) == 0:
                results.append(np.zeros((0, model.get_sentence_embedding_dimension()), dtype=np.float32))
                continue
            windows = [tokenizer.decode(w, skip_special_tokens=True) for w in windows_ids]
            emb = model.encode(windows, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False)
            results.append(emb.astype(np.float32))

    return results

def _model_supported_has_tokenizer_pooling(model, tokenizer):
    # Heuristic: some models might have a pooling head or method
    return hasattr(model, "get_sentence_embedding") or hasattr(model, "pooler")

def make_transformer_extractor(model_id: str) -> Callable[[List[str], Any], List[np.ndarray]]:
    """
    Factory that returns a function embed_texts(texts, cfg) -> list[np.ndarray]
    """
    # We'll delay model loading until first call to embed to allow skipping on load errors
    loaded = {"model": None, "tokenizer": None, "init_error": None}

    def embed_texts(texts: List[str], cfg):
        if loaded["init_error"] is not None:
            raise RuntimeError(loaded["init_error"])
        if loaded["model"] is None:
            try:
                model, tokenizer = _try_model_load(model_id, cfg)
                loaded["model"] = model
                loaded["tokenizer"] = tokenizer
            except Exception as e:
                loaded["init_error"] = e
                raise

        model = loaded["model"]
        tokenizer = loaded["tokenizer"]

        # choose mode: sentence / sliding_windows / prefix / token
        mode = cfg.DEFAULT_TIME_MODE
        # For encoder-decoder name 'bart' or 't5', we treat specially: use encoder hidden states
        # For BART/T5 we will forward through model.encoder if available
        is_encoder_decoder = False
        try:
            # Some models have attribute 'encoder' (bart, t5)
            if hasattr(model, "encoder"):
                is_encoder_decoder = True
        except Exception:
            is_encoder_decoder = False

        # For bart/t5 we will use token-level or sliding windows but ensure we call model.encoder
        # handle token-level specially
        results = _embed_with_transformers_token_windows(model, tokenizer, texts, cfg, mode=mode)
        return results

    return embed_texts

def make_sentence_transformer_extractor(model_id: str) -> Callable[[List[str], Any], List[np.ndarray]]:
    loaded = {"init_error": None}
    def embed_texts(texts: List[str], cfg):
        try:
            return _embed_with_sentence_transformer(model_id, texts, cfg)
        except Exception as e:
            raise

    return embed_texts

# Factory to return extractor given model id
def get_extractor(model_id: str) -> Callable[[List[str], Any], List[np.ndarray]]:
    model_id = model_id.strip()
    if model_id.startswith("sentence-transformers/"):
        return make_sentence_transformer_extractor(model_id)
    else:
        return make_transformer_extractor(model_id)
