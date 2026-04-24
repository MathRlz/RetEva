"""Cache configuration."""
from dataclasses import dataclass
from typing import Optional


@dataclass
class CacheConfig:
    """
    Configuration for caching expensive computations.
    
    The cache stores intermediate results to disk to avoid redundant computation
    across evaluation runs. Different cache types can be individually enabled/disabled.
    
    Attributes:
        enabled: Whether caching is enabled globally. Default: True.
        cache_dir: Directory path for cache storage. Default: ".cache".
        max_size_gb: Optional maximum cache size in GB. If None, unlimited.
        cache_asr_features: Cache preprocessed ASR features. Default: True.
        cache_transcriptions: Cache ASR transcription results. Default: True.
        cache_embeddings: Cache text embedding vectors. Default: True.
        cache_vector_db: Cache built vector databases. Default: True.
    
    Examples:
        >>> config = CacheConfig(enabled=True, cache_dir=".cache")
        >>> config.cache_asr_features = False  # Disable ASR feature caching
        >>> config.max_size_gb = 50.0  # Limit cache to 50GB
    """
    enabled: bool = True
    cache_dir: str = ".cache"
    max_size_gb: Optional[float] = None
    
    cache_asr_features: bool = True
    cache_transcriptions: bool = True
    cache_embeddings: bool = True
    cache_vector_db: bool = True
