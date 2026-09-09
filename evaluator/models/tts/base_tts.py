"""Shared base + helpers for TTS providers: config storage, the synthesize interface,
the torch/transformers import guard, and sample-rate extraction — kept separate so
providers with different backends (Coqui XTTS, Piper) aren't forced into one loading template.
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

    def to(self, device: str) -> "BaseTTSModel":
        """No-op default for providers with no GPU concept (piper: an external CPU-only
        subprocess binary) — a provider that DOES support a device (m4t, mms: transformers;
        xtts_v2: Coqui TTS) overrides this to actually move its model, so callers can treat
        every provider uniformly."""
        return self


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
