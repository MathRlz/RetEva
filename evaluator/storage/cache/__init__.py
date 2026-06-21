"""Caching infrastructure for expensive computations.

Split into focused submodules (``manager`` / ``manifest`` / ``io`` / ``eviction``)
while preserving the original ``evaluator.storage.cache`` public surface.
"""

from ..cache_keys import (
    model_key,
    embedding_key,
    transcription_key,
    audio_embedding_key,
)
from .io import _file_checksum
from .manager import CacheManager, logger
from .manifest import CacheManifestError, _MANIFEST_SCHEMA_VERSION

__all__ = [
    "CacheManager",
    "CacheManifestError",
    "_MANIFEST_SCHEMA_VERSION",
    "_file_checksum",
    "logger",
    "model_key",
    "embedding_key",
    "transcription_key",
    "audio_embedding_key",
]
