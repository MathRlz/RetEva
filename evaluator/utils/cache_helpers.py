"""
Cache helper utilities for pipeline classes.

Provides a mixin class and utilities for consistent caching behavior
across ASR, text embedding, and audio embedding pipelines.
"""

import hashlib
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

import numpy as np
import torch

from ..storage.cache import CacheManager
from ..logging_config import get_logger

logger = get_logger(__name__)

T = TypeVar('T')


def compute_hash(data: bytes, *extra: Any) -> str:
    """
    Compute MD5 hash of bytes data with optional extra components.
    
    Args:
        data: Primary bytes to hash
        *extra: Additional values to include in hash (converted to str then bytes)
        
    Returns:
        MD5 hex digest string
    """
    content = data
    for item in extra:
        content += str(item).encode()
    return hashlib.md5(content).hexdigest()


def compute_audio_hash(audio: torch.Tensor | np.ndarray, sampling_rate: int) -> str:
    """
    Compute a hash for audio data including sampling rate.
    
    Args:
        audio: Audio tensor or numpy array
        sampling_rate: Audio sampling rate
        
    Returns:
        MD5 hex digest string
    """
    if isinstance(audio, torch.Tensor):
        audio_bytes = audio.cpu().numpy().tobytes()
    else:
        audio_bytes = np.asarray(audio).tobytes()
    return compute_hash(audio_bytes, sampling_rate)


class CacheMixin:
    """
    Mixin class providing consistent caching functionality for pipelines.
    
    Pipelines using this mixin should:
    1. Set self.cache to a CacheManager instance (or None)
    2. Call _init_cache_stats() with category names in __init__
    3. Use _check_cache() and _store_cache() for cache operations
    4. Use get_cache_stats() to retrieve statistics
    
    Example:
        class MyPipeline(CacheMixin):
            def __init__(self, model, cache_manager=None):
                self.cache = cache_manager
                self._init_cache_stats(['embeddings'])
    """
    
    cache: Optional[CacheManager]
    _cache_hits: Dict[str, int]
    _cache_misses: Dict[str, int]
    
    def _init_cache_stats(self, categories: List[str]) -> None:
        """
        Initialize cache statistics tracking.
        
        Args:
            categories: List of category names to track (e.g., ['features', 'transcriptions'])
        """
        self._cache_hits = {cat: 0 for cat in categories}
        self._cache_misses = {cat: 0 for cat in categories}
    
    @property
    def cache_enabled(self) -> bool:
        """Check if caching is enabled."""
        return self.cache is not None and self.cache.enabled
    
    def _check_cache(
        self,
        category: str,
        getter: Callable[[], Optional[T]],
        log_key: Optional[str] = None
    ) -> Tuple[Optional[T], bool]:
        """
        Check cache for a value and update statistics.
        
        Args:
            category: Cache category for stats tracking
            getter: Callable that retrieves from cache (returns None if not found)
            log_key: Optional key identifier for debug logging
            
        Returns:
            Tuple of (cached_value or None, was_hit)
        """
        if not self.cache_enabled:
            self._cache_misses[category] += 1
            return None, False
            
        cached_value = getter()
        if cached_value is not None:
            self._cache_hits[category] += 1
            if log_key:
                logger.debug(f"{category} cache hit for {log_key}")
            return cached_value, True
        
        self._cache_misses[category] += 1
        return None, False
    
    def _store_cache(self, setter: Callable[[], None]) -> None:
        """
        Store a value in cache if caching is enabled.
        
        Args:
            setter: Callable that stores the value in cache
        """
        if self.cache_enabled:
            setter()
    
    def _record_hit(self, category: str, count: int = 1) -> None:
        """Record cache hit(s) for a category."""
        self._cache_hits[category] += count
    
    def _record_miss(self, category: str, count: int = 1) -> None:
        """Record cache miss(es) for a category."""
        self._cache_misses[category] += count
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics in a consistent format.
        
        Returns:
            Dictionary with stats per category. Each category has:
            - hits: Number of cache hits
            - misses: Number of cache misses
            - hit_rate: Ratio of hits to total requests (0 if no requests)
            
            If only one category exists, returns flat stats without nesting.
        """
        stats = {}
        for category in self._cache_hits:
            hits = self._cache_hits[category]
            misses = self._cache_misses[category]
            total = hits + misses
            stats[category] = {
                'hits': hits,
                'misses': misses,
                'hit_rate': hits / total if total > 0 else 0.0
            }
        
        # For single-category caches, return flat structure
        if len(stats) == 1:
            return stats[list(stats.keys())[0]]
        
        return stats
    
    def reset_cache_stats(self) -> None:
        """Reset all cache statistics to zero."""
        for category in self._cache_hits:
            self._cache_hits[category] = 0
            self._cache_misses[category] = 0
