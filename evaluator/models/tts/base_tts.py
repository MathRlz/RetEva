"""Shared base + helpers for TTS providers.

Keeps the common bits (config storage, the synthesize interface, the
torch/transformers import guard, sample-rate extraction) in one place without
forcing providers with different backends (Coqui XTTS, Piper) into a single
loading template.
"""
import logging

import numpy as np

logger = logging.getLogger(__name__)


class BaseTTSModel:
    """Common interface for text-to-speech providers."""

    def __init__(self, config):
        self.config = config

    def synthesize(self, text: str) -> np.ndarray:
        """Synthesize ``text`` into a float32 mono waveform."""
        raise NotImplementedError


def require_torch_transformers(provider_label: str):
    """Import torch (and ensure transformers is present) or raise a clear error.

    Returns the imported ``torch`` module so callers can keep a handle to it.
    """
    try:
        import torch
        import transformers  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            f"{provider_label} requires transformers + torch. "
            "Install with: pip install transformers torch"
        ) from exc
    return torch


def model_sampling_rate(model, default: int = 16000) -> int:
    """Read ``model.config.sampling_rate``, falling back to ``default``."""
    return int(getattr(model.config, "sampling_rate", default))
