"""Constants module for the evaluator package.

This module centralizes magic values and hardcoded numbers for better
readability and maintainability.
"""

# =============================================================================
# Numerical Constants
# =============================================================================

# Used for vector normalization to avoid division by zero
MIN_NORM_THRESHOLD = 1e-12

# =============================================================================
# Audio Constants
# =============================================================================

# Standard audio sample rate in Hz (used by most audio models)
DEFAULT_SAMPLE_RATE = 16000

# =============================================================================
# Embedding Dimensions
# =============================================================================

# Text embedding model dimensions
LABSE_DIM = 768
JINA_V4_DIM = 1024
BGE_M3_DIM = 1024
NEMOTRON_DIM = 1024
CLIP_DIM = 768

# Audio embedding dimensions
DEFAULT_AUDIO_EMB_DIM = 1024

# =============================================================================
# Batch Sizes
# =============================================================================

DEFAULT_BATCH_SIZE = 32

# =============================================================================
# FAISS Index Parameters
# =============================================================================

# Number of clusters for IVF index
DEFAULT_FAISS_NLIST = 1024
