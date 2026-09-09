"""Constants module for the evaluator package.

This module centralizes magic values and hardcoded numbers for better
readability and maintainability.
"""

# Used for vector normalization to avoid division by zero
MIN_NORM_THRESHOLD = 1e-12

# Standard audio sample rate in Hz (used by most audio models)
DEFAULT_SAMPLE_RATE = 16000

# Text embedding model dimensions
LABSE_DIM = 768
JINA_V4_DIM = 1024
BGE_M3_DIM = 1024
NEMOTRON_DIM = 1024
CLIP_DIM = 768

# Audio embedding dimensions
DEFAULT_AUDIO_EMB_DIM = 1024

DEFAULT_BATCH_SIZE = 32

# Number of clusters for IVF index
DEFAULT_FAISS_NLIST = 1024

# The rank cutoffs the report always publishes (recall/precision/ndcg @ each). The retriever
# must return at least the deepest one, else `retrieved[:10]` is a `[:k]` slice in disguise and
# `@10` silently equals `@k` — the report would print a cutoff it never measured.
IR_CUTOFFS = (1, 5, 10)
RETRIEVAL_DEPTH = max(IR_CUTOFFS)

# Reducer names accepted by `variant_rollup:` — the single source for both the config check
# and the reducer table in evaluation/aggregate.py, which is keyed by this tuple.
ROLLUP_REDUCERS = ("mean", "min", "max")
