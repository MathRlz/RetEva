"""Cache configuration."""
from dataclasses import dataclass
from typing import Optional


@dataclass
class CacheConfig:
    """Configuration for caching expensive computations to disk, to avoid redundant work
    across evaluation runs."""
    enabled: bool = True
    cache_dir: str = ".cache"
    max_size_gb: Optional[float] = None  # None = unlimited

    cache_asr_features: bool = True
    cache_transcriptions: bool = True
    cache_embeddings: bool = True
    cache_vector_db: bool = True
