"""
Configuration file for the embedding pipeline.
Edit values here — no hardcoded values in the codebase.
"""

import os

# Input / output
INPUT_JSON_PATH = "data/generated_text.json"  # JSON [] array of strings
OUTPUT_DIR = "outputs"  # directory to write .pt and _meta.json files

# HuggingFace token (optional). If set to None, code will still try to load public models.
# Recommended: set HF_TOKEN in environment and refer here with os.environ.get("HF_TOKEN")
HF_TOKEN = os.environ.get("HF_TOKEN", None)

# Model list (order matters for outputs). Use HF ids.
EMBEDDING_METHODS = [
    # Sentence level encoders
    "sentence-transformers/all-mpnet-base-v2",
    "intfloat/e5-large-v2",
    "facebook/contriever",
    # "Alibaba-NLP/gte-Qwen2-7B-instruct",
    # "BAAI/bge-large-en-v1.5",
    # Token level encoders
    # "facebook/bart-large", # token level
    # "t5-base", # token level
]

# Device setup: 'auto' -> will select cuda if available else cpu
DEVICE = "auto"

# Pooling & behavior settings (model-specific pooling defaults in code will try model-provided pooling first)
POOLING = "auto"  # 'auto' -> try model-provided pooling, fallback to mean pooling

# Time/temporal representation options (configurable per run)
# Mode choices: "sliding_windows", "prefix", "sentence", "token"
# You can set a default mode here; it is applied to all models unless overridden programmatically.
DEFAULT_TIME_MODE = "sentence" # "sliding_windows"

# Sliding window parameters (token-level windows)
WINDOW_SIZE_TOKENS = 16
WINDOW_STRIDE_TOKENS = 1

# Prefix parameters: embed cumulative prefixes with steps of this many tokens
PREFIX_STEP_TOKENS = 8

# Sentence splitting option (English only). If True, sentence-level mode uses nltk.sent_tokenize.
SENTENCE_SPLIT = True

# Padding / truncation policy:
# "pad" -> pad sequences to max length per-method
PADDING_POLICY = "truncate" # "pad" or "truncate"
MAX_LENGTH_TRUNCATE = 1024  # only used if PADDING_POLICY == "truncate"

# Pooling normalization and dtype
L2_NORMALIZE = False  # default: no normalization (you asked default off)
DTYPE = "float32"  # "float32" or "float16"

# Batch sizes and memory controls (tweak as needed)
BATCH_SIZE = 32        # general batch size for sentence-level / moderate models
LARGE_MODEL_BATCH_SIZE = 8  # for large models like qwen, bge if running on limited GPU

# Trust remote code when loading model (required for some third-party HF repos)
TRUST_REMOTE_CODE = True

# Behavior on load error:
# "skip" -> print error and skip that model
# "raise" -> raise exception immediately
ON_LOAD_FAIL = "skip"

# Tokenizer & truncation settings (pair with model tokenizer by default)
TOKENIZER_PADDING = "longest"  # used when batching windows/prefixes; can be 'max_length' etc
TOKENIZER_TRUNCATION = True

# Output naming
OUTPUT_FILENAME_TEMPLATE = "{model_slug}.pt"
META_FILENAME_TEMPLATE = "{model_slug}_meta.json"

# Reproducibility / deterministic
SEED = 42

# Other minor options
PRINT_ONLY_MESSAGES = True  # if True print progress messages (no tqdm)
