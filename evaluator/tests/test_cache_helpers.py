"""Tests for cache helper utilities."""

import numpy as np
import pytest
import torch

from evaluator.utils.cache_helpers import CacheMixin, compute_audio_hash, compute_hash


class TestComputeHash:
    """Tests for compute_hash function."""
    
    def test_compute_hash_basic(self):
        """Test basic hash computation."""
        result = compute_hash(b"test data")
        assert isinstance(result, str)
        assert len(result) == 32  # MD5 hex digest length
    
    def test_compute_hash_deterministic(self):
        """Test that same input produces same hash."""
        data = b"test data"
        assert compute_hash(data) == compute_hash(data)
    
    def test_compute_hash_different_inputs(self):
        """Test that different inputs produce different hashes."""
        assert compute_hash(b"data1") != compute_hash(b"data2")
    
    def test_compute_hash_with_extra_args(self):
        """Test hash computation with extra arguments."""
        base = b"test"
        hash1 = compute_hash(base, 16000)
        hash2 = compute_hash(base, 22050)
        assert hash1 != hash2
    
    def test_compute_hash_extra_args_deterministic(self):
        """Test that extra args produce deterministic hash."""
        assert compute_hash(b"test", 16000, "en") == compute_hash(b"test", 16000, "en")


class TestComputeAudioHash:
    """Tests for compute_audio_hash function."""
    
    def test_compute_audio_hash_tensor(self):
        """Test hash for torch tensor."""
        audio = torch.randn(16000)
        result = compute_audio_hash(audio, 16000)
        assert isinstance(result, str)
        assert len(result) == 32
    
    def test_compute_audio_hash_numpy(self):
        """Test hash for numpy array."""
        audio = np.random.randn(16000)
        result = compute_audio_hash(audio, 16000)
        assert isinstance(result, str)
        assert len(result) == 32
    
    def test_compute_audio_hash_deterministic(self):
        """Test that same audio produces same hash."""
        audio = torch.randn(16000)
        assert compute_audio_hash(audio, 16000) == compute_audio_hash(audio, 16000)
    
    def test_compute_audio_hash_different_sample_rate(self):
        """Test that different sample rates produce different hashes."""
        audio = torch.randn(16000)
        hash1 = compute_audio_hash(audio, 16000)
        hash2 = compute_audio_hash(audio, 22050)
        assert hash1 != hash2
    
    def test_compute_audio_hash_different_audio(self):
        """Test that different audio produces different hashes."""
        audio1 = torch.randn(16000)
        audio2 = torch.randn(16000)
        assert compute_audio_hash(audio1, 16000) != compute_audio_hash(audio2, 16000)


class MockCacheManager:
    """Mock cache manager for testing."""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._cache = {}
    
    def get(self, key: str):
        return self._cache.get(key)
    
    def set(self, key: str, value):
        self._cache[key] = value


class ConcreteCacheMixin(CacheMixin):
    """Concrete implementation of CacheMixin for testing."""
    
    def __init__(self, cache_manager=None, categories=None):
        self.cache = cache_manager
        self._init_cache_stats(categories or ['default'])


class TestCacheMixin:
    """Tests for CacheMixin class."""
    
    def test_init_cache_stats_single_category(self):
        """Test initialization with single category."""
        mixin = ConcreteCacheMixin(categories=['embeddings'])
        assert mixin._cache_hits == {'embeddings': 0}
        assert mixin._cache_misses == {'embeddings': 0}
    
    def test_init_cache_stats_multiple_categories(self):
        """Test initialization with multiple categories."""
        mixin = ConcreteCacheMixin(categories=['features', 'transcriptions'])
        assert mixin._cache_hits == {'features': 0, 'transcriptions': 0}
        assert mixin._cache_misses == {'features': 0, 'transcriptions': 0}
    
    def test_cache_enabled_true(self):
        """Test cache_enabled property when cache is enabled."""
        cache = MockCacheManager(enabled=True)
        mixin = ConcreteCacheMixin(cache_manager=cache)
        assert mixin.cache_enabled is True
    
    def test_cache_enabled_false_disabled(self):
        """Test cache_enabled property when cache is disabled."""
        cache = MockCacheManager(enabled=False)
        mixin = ConcreteCacheMixin(cache_manager=cache)
        assert mixin.cache_enabled is False
    
    def test_cache_enabled_false_no_cache(self):
        """Test cache_enabled property when no cache manager."""
        mixin = ConcreteCacheMixin(cache_manager=None)
        assert mixin.cache_enabled is False
    
    def test_check_cache_hit(self):
        """Test _check_cache with cache hit."""
        cache = MockCacheManager(enabled=True)
        cache.set('test_key', 'cached_value')
        mixin = ConcreteCacheMixin(cache_manager=cache)
        
        value, hit = mixin._check_cache(
            'default',
            lambda: cache.get('test_key')
        )
        
        assert value == 'cached_value'
        assert hit is True
        assert mixin._cache_hits['default'] == 1
        assert mixin._cache_misses['default'] == 0
    
    def test_check_cache_miss(self):
        """Test _check_cache with cache miss."""
        cache = MockCacheManager(enabled=True)
        mixin = ConcreteCacheMixin(cache_manager=cache)
        
        value, hit = mixin._check_cache(
            'default',
            lambda: cache.get('nonexistent')
        )
        
        assert value is None
        assert hit is False
        assert mixin._cache_hits['default'] == 0
        assert mixin._cache_misses['default'] == 1
    
    def test_check_cache_disabled(self):
        """Test _check_cache when cache is disabled."""
        cache = MockCacheManager(enabled=False)
        cache.set('test_key', 'cached_value')
        mixin = ConcreteCacheMixin(cache_manager=cache)
        
        value, hit = mixin._check_cache(
            'default',
            lambda: cache.get('test_key')
        )
        
        assert value is None
        assert hit is False
        assert mixin._cache_misses['default'] == 1
    
    def test_store_cache_enabled(self):
        """Test _store_cache when cache is enabled."""
        cache = MockCacheManager(enabled=True)
        mixin = ConcreteCacheMixin(cache_manager=cache)
        
        mixin._store_cache(lambda: cache.set('new_key', 'new_value'))
        
        assert cache.get('new_key') == 'new_value'
    
    def test_store_cache_disabled(self):
        """Test _store_cache when cache is disabled."""
        cache = MockCacheManager(enabled=False)
        mixin = ConcreteCacheMixin(cache_manager=cache)
        
        mixin._store_cache(lambda: cache.set('new_key', 'new_value'))
        
        assert cache.get('new_key') is None
    
    def test_record_hit(self):
        """Test _record_hit method."""
        mixin = ConcreteCacheMixin()
        mixin._record_hit('default')
        mixin._record_hit('default', 5)
        assert mixin._cache_hits['default'] == 6
    
    def test_record_miss(self):
        """Test _record_miss method."""
        mixin = ConcreteCacheMixin()
        mixin._record_miss('default')
        mixin._record_miss('default', 3)
        assert mixin._cache_misses['default'] == 4
    
    def test_get_cache_stats_single_category(self):
        """Test get_cache_stats with single category returns flat dict."""
        mixin = ConcreteCacheMixin(categories=['embeddings'])
        mixin._record_hit('embeddings', 3)
        mixin._record_miss('embeddings', 1)
        
        stats = mixin.get_cache_stats()
        
        assert stats == {
            'hits': 3,
            'misses': 1,
            'hit_rate': 0.75
        }
    
    def test_get_cache_stats_multiple_categories(self):
        """Test get_cache_stats with multiple categories returns nested dict."""
        mixin = ConcreteCacheMixin(categories=['features', 'transcriptions'])
        mixin._record_hit('features', 2)
        mixin._record_miss('features', 2)
        mixin._record_hit('transcriptions', 4)
        mixin._record_miss('transcriptions', 1)
        
        stats = mixin.get_cache_stats()
        
        assert stats == {
            'features': {
                'hits': 2,
                'misses': 2,
                'hit_rate': 0.5
            },
            'transcriptions': {
                'hits': 4,
                'misses': 1,
                'hit_rate': 0.8
            }
        }
    
    def test_get_cache_stats_no_requests(self):
        """Test get_cache_stats with no cache requests."""
        mixin = ConcreteCacheMixin(categories=['embeddings'])
        
        stats = mixin.get_cache_stats()
        
        assert stats == {
            'hits': 0,
            'misses': 0,
            'hit_rate': 0.0
        }
    
    def test_reset_cache_stats(self):
        """Test reset_cache_stats method."""
        mixin = ConcreteCacheMixin(categories=['features', 'transcriptions'])
        mixin._record_hit('features', 5)
        mixin._record_miss('transcriptions', 3)
        
        mixin.reset_cache_stats()
        
        assert mixin._cache_hits == {'features': 0, 'transcriptions': 0}
        assert mixin._cache_misses == {'features': 0, 'transcriptions': 0}
