"""Unit tests for CacheManager class."""
import json
import sqlite3
import pytest
import tempfile
import shutil
import numpy as np
from pathlib import Path

from evaluator.storage.cache import CacheManager
from evaluator.storage.cache_keys import (
    model_key,
    embedding_key,
    transcription_key,
    dataset_fingerprint,
    preprocessing_fingerprint,
)


class TestCacheManagerInit:
    """Tests for CacheManager initialization."""

    def test_creates_cache_directories_when_enabled(self):
        """Cache directories are created when enabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "test_cache"
            manager = CacheManager(cache_dir=str(cache_dir), enabled=True)

            assert manager.asr_features_dir.exists()
            assert manager.transcriptions_dir.exists()
            assert manager.embeddings_dir.exists()
            assert manager.audio_embeddings_dir.exists()
            assert manager.vector_db_dir.exists()
            assert manager.checkpoints_dir.exists()
            assert manager.manifest_db_path.exists()

    def test_no_directories_when_disabled(self):
        """Cache directories not created when disabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "test_cache"
            manager = CacheManager(cache_dir=str(cache_dir), enabled=False)

            assert not cache_dir.exists()


class TestCacheKeyGeneration:
    """Tests for cache key generation functions."""

    def test_hash_is_deterministic(self):
        """Same inputs produce same hash."""
        hash1 = model_key("audio123", "model_a")
        hash2 = model_key("audio123", "model_a")

        assert hash1 == hash2

    def test_hash_is_unique_for_different_inputs(self):
        """Different inputs produce different hashes."""
        hash1 = model_key("audio123", "model_a")
        hash2 = model_key("audio456", "model_a")
        hash3 = model_key("audio123", "model_b")

        assert hash1 != hash2
        assert hash1 != hash3
        assert hash2 != hash3

    def test_hash_length(self):
        """Hash is 32 characters (MD5 hex digest)."""
        result = model_key("test", "data")
        assert len(result) == 32

    def test_hash_handles_none_values(self):
        """Hash handles None values correctly."""
        hash1 = transcription_key("audio", "model", None)
        hash2 = transcription_key("audio", "model", None)
        hash3 = transcription_key("audio", "model", "en")

        assert hash1 == hash2
        assert hash1 != hash3


class TestCachePath:
    """Tests for _get_cache_path method."""

    def test_valid_cache_types(self):
        """Valid cache types return correct paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CacheManager(cache_dir=tmpdir, enabled=True)

            assert manager._get_cache_path("asr_features", "key1", ".npz").parent == manager.asr_features_dir
            assert manager._get_cache_path("transcriptions", "key2", ".json").parent == manager.transcriptions_dir
            assert manager._get_cache_path("embeddings", "key3", ".npy").parent == manager.embeddings_dir
            assert manager._get_cache_path("audio_embeddings", "key4", ".npy").parent == manager.audio_embeddings_dir
            assert manager._get_cache_path("vector_db", "key5", "").parent == manager.vector_db_dir
            assert manager._get_cache_path("checkpoints", "key6", ".json").parent == manager.checkpoints_dir

    def test_invalid_cache_type_raises_error(self):
        """Invalid cache type raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CacheManager(cache_dir=tmpdir, enabled=True)

            with pytest.raises(ValueError, match="Unknown cache type"):
                manager._get_cache_path("invalid_type", "key", ".ext")


class TestTranscriptionCache:
    """Tests for transcription caching."""

    def test_cache_miss_returns_none(self):
        """Cache miss returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CacheManager(cache_dir=tmpdir, enabled=True)

            result = manager.get_transcription("audio_hash", "model_name")
            assert result is None

    def test_cache_hit_returns_value(self):
        """Cache hit returns stored value."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CacheManager(cache_dir=tmpdir, enabled=True)

            manager.set_transcription("audio_hash", "model_name", "Hello world")
            result = manager.get_transcription("audio_hash", "model_name")

            assert result == "Hello world"

    def test_different_language_separate_cache(self):
        """Different languages have separate cache entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CacheManager(cache_dir=tmpdir, enabled=True)

            manager.set_transcription("audio", "model", "English text", language="en")
            manager.set_transcription("audio", "model", "Polish text", language="pl")

            assert manager.get_transcription("audio", "model", "en") == "English text"
            assert manager.get_transcription("audio", "model", "pl") == "Polish text"
            assert manager.get_transcription("audio", "model") is None

    def test_disabled_cache_returns_none(self):
        """Disabled cache always returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CacheManager(cache_dir=tmpdir, enabled=False)

            manager.set_transcription("audio", "model", "text")
            result = manager.get_transcription("audio", "model")

            assert result is None


class TestEmbeddingCache:
    """Tests for embedding caching."""

    def test_cache_miss_returns_none(self):
        """Cache miss returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CacheManager(cache_dir=tmpdir, enabled=True)

            result = manager.get_embedding("text", "model")
            assert result is None

    def test_cache_hit_returns_embedding(self):
        """Cache hit returns stored embedding."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CacheManager(cache_dir=tmpdir, enabled=True)

            embedding = np.array([0.1, 0.2, 0.3, 0.4])
            manager.set_embedding("test text", "model", embedding)
            result = manager.get_embedding("test text", "model")

            np.testing.assert_array_almost_equal(result, embedding)

    def test_batch_embeddings_all_hit(self):
        """Batch returns all embeddings when all cached."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CacheManager(cache_dir=tmpdir, enabled=True)

            texts = ["text1", "text2", "text3"]
            embeddings = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])

            manager.set_embeddings_batch(texts, "model", embeddings)
            result = manager.get_embeddings_batch(texts, "model")

            np.testing.assert_array_almost_equal(result, embeddings)

    def test_batch_embeddings_partial_miss_returns_none(self):
        """Batch returns None if any embedding is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CacheManager(cache_dir=tmpdir, enabled=True)

            manager.set_embedding("text1", "model", np.array([0.1, 0.2]))
            # text2 not cached

            result = manager.get_embeddings_batch(["text1", "text2"], "model")
            assert result is None


class TestASRFeaturesCache:
    """Tests for ASR features caching."""

    def test_cache_miss_returns_none(self):
        """Cache miss returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CacheManager(cache_dir=tmpdir, enabled=True)

            result = manager.get_asr_features("audio_hash", "model")
            assert result is None

    def test_cache_hit_returns_features(self):
        """Cache hit returns features and attention mask."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CacheManager(cache_dir=tmpdir, enabled=True)

            features = np.random.randn(1, 100, 768).astype(np.float32)
            attention_mask = np.ones((1, 100), dtype=np.float32)

            manager.set_asr_features("audio_hash", "model", features, attention_mask)
            cached_features, cached_mask = manager.get_asr_features("audio_hash", "model")

            np.testing.assert_array_almost_equal(cached_features, features)
            np.testing.assert_array_almost_equal(cached_mask, attention_mask)


class TestAudioEmbeddingCache:
    """Tests for audio embedding caching."""

    def test_cache_hit(self):
        """Cache hit returns stored audio embedding."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CacheManager(cache_dir=tmpdir, enabled=True)

            embedding = np.random.randn(768).astype(np.float32)
            manager.set_audio_embedding("audio_hash", "model", embedding)
            result = manager.get_audio_embedding("audio_hash", "model")

            np.testing.assert_array_almost_equal(result, embedding)

    def test_disabled_returns_none(self):
        """Disabled cache returns None for audio embeddings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CacheManager(cache_dir=tmpdir, enabled=False)

            embedding = np.random.randn(768)
            manager.set_audio_embedding("audio_hash", "model", embedding)
            result = manager.get_audio_embedding("audio_hash", "model")

            assert result is None


class TestVectorDBCache:
    """Tests for vector database caching."""

    def test_cache_miss_returns_none(self):
        """Cache miss returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CacheManager(cache_dir=tmpdir, enabled=True)

            result = manager.get_vector_db("nonexistent_key")
            assert result is None

    def test_cache_hit_returns_vectors_and_texts(self):
        """Cache hit returns vectors and texts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CacheManager(cache_dir=tmpdir, enabled=True)

            vectors = np.random.randn(100, 768).astype(np.float32)
            texts = [f"document_{i}" for i in range(100)]

            manager.set_vector_db("test_db", vectors, texts)
            cached_vectors, cached_texts = manager.get_vector_db("test_db")

            np.testing.assert_array_almost_equal(cached_vectors, vectors)
            assert cached_texts == texts


class TestCheckpointCache:
    """Tests for checkpoint caching."""

    def test_get_nonexistent_checkpoint(self):
        """Getting nonexistent checkpoint returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CacheManager(cache_dir=tmpdir, enabled=True)

            result = manager.get_checkpoint("nonexistent_experiment")
            assert result is None

    def test_set_and_get_checkpoint(self):
        """Setting and getting checkpoint works."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CacheManager(cache_dir=tmpdir, enabled=True)

            checkpoint_data = {
                "last_idx": 50,
                "wer_scores": [0.1, 0.2, 0.15],
                "all_retrieved": [["doc1", "doc2"]],
            }

            manager.set_checkpoint("experiment_1", checkpoint_data)
            result = manager.get_checkpoint("experiment_1")

            assert result == checkpoint_data

    def test_disabled_checkpoint_returns_none(self):
        """Disabled cache returns None for checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CacheManager(cache_dir=tmpdir, enabled=False)

            manager.set_checkpoint("exp", {"data": 1})
            result = manager.get_checkpoint("exp")

            assert result is None


class TestUniqueTextsCache:
    """Tests for unique texts caching."""

    def test_cache_miss(self):
        """Cache miss returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CacheManager(cache_dir=tmpdir, enabled=True)

            result = manager.get_unique_texts("dataset", 1000)
            assert result is None

    def test_cache_hit(self):
        """Cache hit returns stored unique texts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CacheManager(cache_dir=tmpdir, enabled=True)

            texts = ["text1", "text2", "text3"]
            manager.set_unique_texts("my_dataset", 100, texts)
            result = manager.get_unique_texts("my_dataset", 100)

            assert result == texts

    def test_different_size_separate_cache(self):
        """Different dataset sizes have separate cache entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CacheManager(cache_dir=tmpdir, enabled=True)

            manager.set_unique_texts("dataset", 100, ["small"])
            manager.set_unique_texts("dataset", 1000, ["large"])

            assert manager.get_unique_texts("dataset", 100) == ["small"]
            assert manager.get_unique_texts("dataset", 1000) == ["large"]

    def test_manifest_fingerprint_cache_hit(self):
        """Manifest-based unique text cache keying works."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CacheManager(cache_dir=tmpdir, enabled=True)
            d_fp = dataset_fingerprint("pubmed", trace_limit=50, source={"prepared_dir": "/tmp/data"})
            p_fp = preprocessing_fingerprint({"normalize": True})
            texts = ["a", "b"]

            manager.set_unique_texts(
                unique_texts=texts,
                dataset_fingerprint=d_fp,
                preprocessing_fingerprint=p_fp,
            )
            result = manager.get_unique_texts(
                dataset_fingerprint=d_fp,
                preprocessing_fingerprint=p_fp,
            )
            assert result == texts

    def test_manifest_fingerprint_cache_miss_for_different_preprocessing(self):
        """Different preprocessing fingerprint should miss."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CacheManager(cache_dir=tmpdir, enabled=True)
            d_fp = dataset_fingerprint("pubmed")
            p1 = preprocessing_fingerprint({"normalize": True})
            p2 = preprocessing_fingerprint({"normalize": False})

            manager.set_unique_texts(
                unique_texts=["x"],
                dataset_fingerprint=d_fp,
                preprocessing_fingerprint=p1,
            )
            assert manager.get_unique_texts(dataset_fingerprint=d_fp, preprocessing_fingerprint=p2) is None

    def test_unique_texts_requires_valid_identifiers(self):
        """Calling without required key inputs should raise."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CacheManager(cache_dir=tmpdir, enabled=True)

            with pytest.raises(ValueError, match="Provide either dataset_fingerprint"):
                manager.get_unique_texts()
            with pytest.raises(ValueError, match="Provide either dataset_fingerprint"):
                manager.set_unique_texts(unique_texts=["x"])


class TestCacheStats:
    """Tests for cache statistics reporting."""

    def test_get_cache_size_empty(self):
        """Empty cache returns zero sizes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CacheManager(cache_dir=tmpdir, enabled=True)

            sizes = manager.get_cache_size()

            assert sizes["total"] == 0
            assert sizes["embeddings"] == 0
            assert sizes["transcriptions"] == 0

    def test_get_cache_size_with_data(self):
        """Cache with data reports non-zero sizes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CacheManager(cache_dir=tmpdir, enabled=True)

            # Add some data
            manager.set_transcription("audio", "model", "Some long transcription text")
            embedding = np.random.randn(768).astype(np.float32)
            manager.set_embedding("text", "model", embedding)

            sizes = manager.get_cache_size()

            assert sizes["transcriptions"] > 0
            assert sizes["embeddings"] > 0
            assert sizes["total"] > 0

    def test_get_cache_stats(self):
        """Get detailed cache statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CacheManager(cache_dir=tmpdir, enabled=True)

            manager.set_transcription("a", "m", "text1")
            manager.set_transcription("b", "m", "text2")

            stats = manager.get_cache_stats()

            assert "sizes_bytes" in stats
            assert "file_counts" in stats
            assert "sizes_human" in stats
            assert stats["file_counts"]["transcriptions"] == 2

    def test_human_readable_size(self):
        """Human readable size formatting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CacheManager(cache_dir=tmpdir, enabled=True)

            assert manager._human_readable_size(500) == "500.00 B"
            assert manager._human_readable_size(1024) == "1.00 KB"
            assert manager._human_readable_size(1024 * 1024) == "1.00 MB"
            assert manager._human_readable_size(1024 * 1024 * 1024) == "1.00 GB"


class TestCacheInvalidation:
    """Tests for cache invalidation."""

    def test_clear_type(self):
        """Clear specific cache type."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CacheManager(cache_dir=tmpdir, enabled=True)

            manager.set_transcription("a", "m", "text")
            manager.set_embedding("t", "m", np.array([0.1, 0.2]))

            manager.clear_type("transcriptions")

            # Transcriptions cleared
            assert manager.get_transcription("a", "m") is None
            # Embeddings still present
            assert manager.get_embedding("t", "m") is not None

    def test_clear_type_invalid_raises_error(self):
        """Clear invalid cache type raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CacheManager(cache_dir=tmpdir, enabled=True)

            with pytest.raises(ValueError, match="Unknown cache type"):
                manager.clear_type("invalid_type")

    def test_clear_all(self):
        """Clear all caches."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CacheManager(cache_dir=tmpdir, enabled=True)

            manager.set_transcription("a", "m", "text")
            manager.set_embedding("t", "m", np.array([0.1, 0.2]))

            manager.clear_all()

            assert manager.get_transcription("a", "m") is None
            assert manager.get_embedding("t", "m") is None
            # Directories should be recreated
            assert manager.transcriptions_dir.exists()
            assert manager.embeddings_dir.exists()


class TestCacheManifestIndex:
    """Tests for SQLite cache manifest indexing."""

    def test_transcription_cache_entry_written_to_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CacheManager(cache_dir=tmpdir, enabled=True)
            manager.set_transcription("audio-hash", "whisper-small", "hello", language="en")

            with sqlite3.connect(manager.manifest_db_path) as conn:
                row = conn.execute(
                    """
                    SELECT cache_type, stage, model_name, input_hash, config_hash, artifact_path
                    FROM cache_entries
                    WHERE cache_type = 'transcriptions'
                    """
                ).fetchone()

            assert row is not None
            assert row[0] == "transcriptions"
            assert row[1] == "transcription"
            assert row[2] == "whisper-small"
            assert row[3] == "audio-hash"
            assert row[4] == "en"
            assert Path(row[5]).exists()

    def test_clear_type_removes_manifest_entries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CacheManager(cache_dir=tmpdir, enabled=True)
            manager.set_transcription("a", "m", "text")
            manager.clear_type("transcriptions")

            with sqlite3.connect(manager.manifest_db_path) as conn:
                count = conn.execute(
                    "SELECT COUNT(*) FROM cache_entries WHERE cache_type = 'transcriptions'"
                ).fetchone()[0]

            assert count == 0

    def test_clear_model_uses_manifest_model_name(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CacheManager(cache_dir=tmpdir, enabled=True)
            manager.set_transcription("a1", "model-A", "text")
            manager.set_transcription("a2", "model-B", "text")

            cleared = manager.clear_model("model-A")

            assert cleared == 1
            assert manager.get_transcription("a1", "model-A") is None
            assert manager.get_transcription("a2", "model-B") == "text"
