"""Caching infrastructure for expensive computations.

Split into focused submodules (``manager`` / ``manifest`` / ``io`` / ``eviction``)
while preserving the original ``evaluator.storage.cache`` public surface.
"""

from ..cache_keys import (
    model_key,
    embedding_key,
    transcription_key,
    audio_embedding_key,
    unique_texts_key,
    unique_texts_manifest_key,
)
from .io import _file_checksum
from .manager import CacheManager, logger

__all__ = [
    "CacheManager",
    "_file_checksum",
    "logger",
    "model_key",
    "embedding_key",
    "transcription_key",
    "audio_embedding_key",
    "unique_texts_key",
    "unique_texts_manifest_key",
]
