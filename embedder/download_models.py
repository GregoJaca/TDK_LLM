"""Utility to pre-install Python packages (via pip) and pre-download model weights/tokenizers.

Usage:
    python download_models.py        # will pip install requirements.txt and download models from config.EMBEDDING_METHODS
    python download_models.py --no-pip  # skip pip install

This script intentionally does not force package versions (lets pip resolve), per project request.
It will attempt to load each model listed in `config.EMBEDDING_METHODS` and its tokenizer.
For `sentence-transformers/*` models it will call SentenceTransformer to cache files.
For transformers models it will load AutoTokenizer and AutoModel (in cpu) to cache files.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import os
import importlib
from pathlib import Path
from typing import List


def pip_install(req_path: Path) -> None:
    print(f"Installing packages from {req_path} using pip (this may take a while)...")
    cmd = [sys.executable, "-m", "pip", "install", "-r", str(req_path)]
    subprocess.check_call(cmd)


def ensure_import(name: str):
    try:
        return importlib.import_module(name)
    except Exception as e:
        raise


def cache_models(models: List[str], cfg_module_path: str = "config"):
    # Import config to get TOKEN if present
    cfg = importlib.import_module(cfg_module_path)

    hf_token = getattr(cfg, "HF_TOKEN", None)
    # set HF token into env for requests if present
    if hf_token:
        os.environ.setdefault("HF_TOKEN", hf_token)

    # Lazy imports
    from transformers import AutoTokenizer, AutoModel
    from sentence_transformers import SentenceTransformer

    for m in models:
        print(f"Caching model: {m}")
        try:
            if m.startswith("sentence-transformers/"):
                # SentenceTransformer will download tokenizer + model
                SentenceTransformer(m, device="cpu")
            else:
                # load tokenizer and model on CPU
                AutoTokenizer.from_pretrained(m, use_fast=True)
                AutoModel.from_pretrained(m)
        except Exception as e:
            print(f"Warning: failed to fully cache {m}: {e}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--no-pip", action="store_true", help="Do not run pip install; only download models")
    args = p.parse_args()

    root = Path(__file__).parent
    req = root / "requirements.txt"
    if not args.no_pip:
        if req.exists():
            try:
                pip_install(req)
            except subprocess.CalledProcessError as e:
                print(f"pip install failed: {e}; you may need to run it manually and ensure packages are available.")
        else:
            print("No requirements.txt found; skipping pip install.")

    # Now import config to read EMBEDDING_METHODS
    try:
        import config as cfg
    except Exception as e:
        print(f"Failed to import config.py: {e}")
        raise

    models = getattr(cfg, "EMBEDDING_METHODS", [])
    if not models:
        print("No models listed in config.EMBEDDING_METHODS; nothing to download.")
        return

    try:
        cache_models(models)
    except Exception as e:
        print(f"Encountered error while caching models: {e}")

    print("Done. Models attempted to be cached into Hugging Face cache.")


if __name__ == "__main__":
    main()
