"""Base utilities and constants for configuration management."""
import re
from typing import Optional, Dict

from ..logging_config import get_logger

logger = get_logger(__name__)

# Device string validation pattern
DEVICE_PATTERN = re.compile(r'^(cpu|cuda(:\d+)?|mps)$')


def validate_device_string(device: str) -> None:
    """Validate device string format.

    Args:
        device: Device string to validate (e.g., "cpu", "cuda:0", "mps").

    Raises:
        ValueError: If device string format is invalid.
    """
    if not DEVICE_PATTERN.match(device):
        raise ValueError(
            f"Invalid device format: '{device}'. "
            f"Expected 'cpu', 'cuda', 'cuda:N', or 'mps'."
        )


# Model size estimates in GB (approximate VRAM requirements: weights at their load dtype
# plus inference activation headroom). Under-estimating packs too many models on one GPU
# and the run dies with a runtime OOM the pool never saw coming — when in doubt, round UP:
# an over-estimate merely spreads models across devices or parks one on CPU.
MODEL_MEMORY_ESTIMATES_GB: Dict[str, Dict[str, float]] = {
    "asr": {
        "whisper": 3.5,  # whisper-large fp16 + decode activations (medium fits in less)
        "wav2vec2": 1.5,
        "default": 2.5,
    },
    "text_embedding": {
        "labse": 2.0,
        "jina_v4": 9.0,  # 3.8B backbone + long-document activation peaks
        "bge_m3": 2.5,
        "nemotron": 3.0,
        "clip": 1.5,
        "eam_alignment": 6.0,  # SeamlessM4T-v2 text encoder fp32 + projection head
        "default": 3.0,
    },
    "audio_embedding": {
        "attention_pool": 4.0,  # whisper-large / M4T speech encoder fp32 + APM head
        "clap_style": 2.0,
        "eam_alignment": 6.0,  # SeamlessM4T-v2 speech encoder fp32 + projection head
        "default": 3.0,
    },
    # TTS providers (keyed by provider name). A 0.0 entry marks a CPU-only provider the
    # pool should not reserve GPU memory for (piper runs as an external subprocess binary).
    "tts": {
        "xtts_v2": 3.0,  # Coqui XTTS-v2 fp32 + vocoder
        "m4t": 10.0,  # full SeamlessM4T-v2-large fp32 (encoder+decoder+t2u+vocoder)
        "mms": 1.0,  # VITS-style, small
        "piper": 0.0,
        "default": 3.0,
    },
}

# Known embedding dimensions for model types
MODEL_EMBEDDING_DIMS: Dict[str, int] = {
    "labse": 768,
    "jina_v4": 2048,
    "bge_m3": 1024,
    "nemotron": 1024,
    "clip": 768,
    "sonar": 1024,
}


def detect_device(preferred: Optional[str] = None) -> str:
    """Detect the best available device for model execution.

    Args:
        preferred: Preferred device string (e.g., "cuda:0", "cpu"). If valid, use it.

    Returns:
        Device string: the preferred device if valid, otherwise auto-detected device.

    Raises:
        ValueError: If preferred device has invalid format.
    """
    import torch

    # If preferred is specified, validate and return it
    if preferred is not None:
        # Validate format first
        validate_device_string(preferred)

        if preferred == "cpu":
            return "cpu"
        if preferred.startswith("cuda"):
            if torch.cuda.is_available():
                # Check if the specific CUDA device exists
                try:
                    device_idx = int(preferred.split(":")[1]) if ":" in preferred else 0
                    if device_idx < torch.cuda.device_count():
                        return preferred
                except (ValueError, IndexError):
                    pass
            # Preferred CUDA device not available, fall through to auto-detect

    # Auto-detect
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def get_available_gpu_count() -> int:
    """Return the number of available CUDA GPUs.

    Returns:
        Number of CUDA devices available, 0 if CUDA is not available.
    """
    try:
        import torch

        return torch.cuda.device_count() if torch.cuda.is_available() else 0
    except ImportError:
        return 0


def get_gpu_memory_gb(device_idx: int = 0) -> Optional[float]:
    """Get total GPU memory in GB for a specific device.

    Args:
        device_idx: CUDA device index.

    Returns:
        Total memory in GB, or None if unavailable.
    """
    try:
        import torch
        if torch.cuda.is_available() and device_idx < torch.cuda.device_count():
            props = torch.cuda.get_device_properties(device_idx)
            return props.total_memory / (1024 ** 3)
    except (ImportError, RuntimeError, ValueError, OSError, AttributeError):
        return None
    return None


def get_gpu_free_memory_gb(device_idx: int = 0) -> Optional[float]:
    """Get free GPU memory in GB for a specific device.

    Args:
        device_idx: CUDA device index.

    Returns:
        Free memory in GB, or None if unavailable.
    """
    try:
        import torch
        if torch.cuda.is_available() and device_idx < torch.cuda.device_count():
            free, total = torch.cuda.mem_get_info(device_idx)
            return free / (1024 ** 3)
    except (ImportError, RuntimeError, ValueError, OSError, AttributeError):
        return None
    return None


def estimate_model_memory_gb(
    model_category: str,
    model_type: Optional[str]
) -> float:
    """Estimate VRAM requirements for a model.

    Args:
        model_category: One of 'asr', 'text_embedding', 'audio_embedding'.
        model_type: The specific model type (e.g., 'whisper', 'labse').

    Returns:
        Estimated memory in GB.
    """
    if model_type is None:
        return 0.0

    category_estimates = MODEL_MEMORY_ESTIMATES_GB.get(model_category, {})
    if model_type not in category_estimates:
        # F26: a genuinely-unregistered model silently borrows the category default, which
        # skews GPU-pool packing. Surface it so the estimate map gets the new entry.
        logger.warning(
            "No memory estimate for %s model '%s'; using the category default "
            "(%.1f GB). Add it to MODEL_MEMORY_ESTIMATES_GB['%s'] for accurate packing.",
            model_category, model_type,
            category_estimates.get("default", 1.5), model_category,
        )
    return category_estimates.get(model_type, category_estimates.get("default", 1.5))


def get_text_embedding_dim(model_type: Optional[str]) -> Optional[int]:
    """Get the embedding dimension for a text embedding model type.

    Args:
        model_type: The text embedding model type.

    Returns:
        Embedding dimension, or None if unknown.
    """
    if model_type is None:
        return None
    if model_type not in MODEL_EMBEDDING_DIMS:
        # F26: unknown dim — the caller falls back to runtime probing; log so a missing
        # MODEL_EMBEDDING_DIMS entry for a known model is visible.
        logger.debug(
            "No embedding dim registered for text model '%s'; resolving at runtime.",
            model_type,
        )
    return MODEL_EMBEDDING_DIMS.get(model_type)
