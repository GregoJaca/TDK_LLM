# Embedding Pipeline (model-agnostic)

This repo implements a flexible pipeline to encode multiple generated text responses
(trajectories) into time-varying embeddings suitable for divergence / Lyapunov-style analysis.

## Overview
- Input: JSON file containing a top-level array of strings: `["resp1...", "resp2...", ...]`.
- Output: For each model in `config.EMBEDDING_METHODS`, a `.pt` file with a tensor of shape
  `[n_traj, len_traj, emb_dim]` and a metadata JSON file with lengths and config.
- Supports multiple temporal modes: sentence-level, sliding windows, prefix-based, token-level.
- Tries to use model-provided pooling if available; otherwise falls back to mean-pooling.
- Loads models using Hugging Face Transformers / sentence-transformers.

## Quick start
1. Edit `config.py` to set `INPUT_JSON_PATH`, `EMBEDDING_METHODS` and other parameters.
2. (Optional) Set `HF_TOKEN` environment variable or put token value in `config.HF_TOKEN`.
3. Run:
   ```bash
   python embedding_pipeline.py
