"""Tests for cache key generation module."""
import pytest
from evaluator.storage.cache_keys import (
    CACHE_SCHEMA_VERSION,
    model_key,
    embedding_key,
    transcription_key,
    audio_embedding_key,
    vector_db_key,
    unique_texts_key,
    manifest_fingerprint,
    dataset_fingerprint,
    model_fingerprint,
    retrieval_fingerprint,
    preprocessing_fingerprint,
    vector_db_manifest_key,
    unique_texts_manifest_key,
)


class TestModelKey:
    """Tests for model_key function."""
    
    def test_deterministic(self):
        """Same inputs produce same key."""
        key1 = model_key("audio123", "model_a")
        key2 = model_key("audio123", "model_a")
        assert key1 == key2
    
    def test_unique_for_different_audio(self):
        """Different audio hashes produce different keys."""
        key1 = model_key("audio123", "model_a")
        key2 = model_key("audio456", "model_a")
        assert key1 != key2
    
    def test_unique_for_different_model(self):
        """Different model names produce different keys."""
        key1 = model_key("audio123", "model_a")
        key2 = model_key("audio123", "model_b")
        assert key1 != key2


class TestEmbeddingKey:
    """Tests for embedding_key function."""
    
    def test_deterministic(self):
        """Same inputs produce same key."""
        key1 = embedding_key("hello world", "labse")
        key2 = embedding_key("hello world", "labse")
        assert key1 == key2
    
    def test_unique_for_different_text(self):
        """Different texts produce different keys."""
        key1 = embedding_key("hello", "labse")
        key2 = embedding_key("world", "labse")
        assert key1 != key2
    
    def test_unique_for_different_model(self):
        """Different models produce different keys."""
        key1 = embedding_key("hello", "labse")
        key2 = embedding_key("hello", "jina")
        assert key1 != key2


class TestTranscriptionKey:
    """Tests for transcription_key function."""
    
    def test_deterministic(self):
        """Same inputs produce same key."""
        key1 = transcription_key("audio123", "whisper", "en")
        key2 = transcription_key("audio123", "whisper", "en")
        assert key1 == key2
    
    def test_language_affects_key(self):
        """Different languages produce different keys."""
        key1 = transcription_key("audio123", "whisper", "en")
        key2 = transcription_key("audio123", "whisper", "pl")
        assert key1 != key2
    
    def test_none_language_differs_from_explicit(self):
        """None language produces different key than explicit language."""
        key1 = transcription_key("audio123", "whisper", None)
        key2 = transcription_key("audio123", "whisper", "en")
        assert key1 != key2
    
    def test_same_with_none_language(self):
        """None language is deterministic."""
        key1 = transcription_key("audio123", "whisper", None)
        key2 = transcription_key("audio123", "whisper", None)
        assert key1 == key2


class TestAudioEmbeddingKey:
    """Tests for audio_embedding_key function."""
    
    def test_deterministic(self):
        """Same inputs produce same key."""
        key1 = audio_embedding_key("audio123", "clap")
        key2 = audio_embedding_key("audio123", "clap")
        assert key1 == key2
    
    def test_unique_for_different_audio(self):
        """Different audio hashes produce different keys."""
        key1 = audio_embedding_key("audio123", "clap")
        key2 = audio_embedding_key("audio456", "clap")
        assert key1 != key2


class TestVectorDbKey:
    """Tests for vector_db_key function."""
    
    def test_deterministic(self):
        """Same inputs produce same key."""
        key1 = vector_db_key("pubmed", 1000, "labse")
        key2 = vector_db_key("pubmed", 1000, "labse")
        assert key1 == key2
    
    def test_unique_for_different_dataset(self):
        """Different datasets produce different keys."""
        key1 = vector_db_key("pubmed", 1000, "labse")
        key2 = vector_db_key("common_voice", 1000, "labse")
        assert key1 != key2
    
    def test_unique_for_different_size(self):
        """Different dataset sizes produce different keys."""
        key1 = vector_db_key("pubmed", 1000, "labse")
        key2 = vector_db_key("pubmed", 2000, "labse")
        assert key1 != key2
    
    def test_unique_for_different_model(self):
        """Different models produce different keys."""
        key1 = vector_db_key("pubmed", 1000, "labse")
        key2 = vector_db_key("pubmed", 1000, "jina")
        assert key1 != key2


class TestUniqueTextsKey:
    """Tests for unique_texts_key function."""
    
    def test_deterministic(self):
        """Same inputs produce same key."""
        key1 = unique_texts_key("pubmed", 5000)
        key2 = unique_texts_key("pubmed", 5000)
        assert key1 == key2
    
    def test_unique_for_different_dataset(self):
        """Different datasets produce different keys."""
        key1 = unique_texts_key("pubmed", 5000)
        key2 = unique_texts_key("common_voice", 5000)
        assert key1 != key2
    
    def test_unique_for_different_size(self):
        """Different sizes produce different keys."""
        key1 = unique_texts_key("pubmed", 5000)
        key2 = unique_texts_key("pubmed", 10000)
        assert key1 != key2


class TestKeyFormat:
    """Tests for key format and properties."""
    
    def test_keys_are_strings(self):
        """All keys are strings."""
        key1 = model_key("audio", "model")
        key2 = embedding_key("text", "model")
        key3 = transcription_key("audio", "model", "en")
        
        assert isinstance(key1, str)
        assert isinstance(key2, str)
        assert isinstance(key3, str)
    
    def test_keys_are_hexadecimal(self):
        """All keys are valid hexadecimal strings (MD5 format)."""
        key = model_key("audio", "model")
        
        # MD5 hashes are 32 characters long
        assert len(key) == 32
        
        # All characters should be hex digits
        assert all(c in '0123456789abcdef' for c in key.lower())


class TestManifestFingerprints:
    """Tests for manifest-based fingerprint and keys."""

    def test_manifest_fingerprint_stable_for_dict_order(self):
        m1 = {"a": 1, "b": {"x": 2, "y": [1, 2, 3]}}
        m2 = {"b": {"y": [1, 2, 3], "x": 2}, "a": 1}
        assert manifest_fingerprint(m1) == manifest_fingerprint(m2)

    def test_schema_version_changes_fingerprint(self):
        payload = {"x": 1}
        key_v1 = manifest_fingerprint(payload, schema_version="v1")
        key_v2 = manifest_fingerprint(payload, schema_version="v2")
        assert key_v1 != key_v2

    def test_specialized_fingerprints_are_deterministic(self):
        d1 = dataset_fingerprint("pubmed", trace_limit=10, source={"prepared_dir": "/a"})
        d2 = dataset_fingerprint("pubmed", trace_limit=10, source={"prepared_dir": "/a"})
        assert d1 == d2

        m1 = model_fingerprint("labse", model_type="sentence_transformer", inference={"device": "cpu"})
        m2 = model_fingerprint("labse", model_type="sentence_transformer", inference={"device": "cpu"})
        assert m1 == m2

        r1 = retrieval_fingerprint(
            vector_store_type="faiss",
            retrieval_strategy={"core": {"mode": "dense"}, "reranking": {"mode": "none"}},
        )
        r2 = retrieval_fingerprint(
            vector_store_type="faiss",
            retrieval_strategy={"reranking": {"mode": "none"}, "core": {"mode": "dense"}},
        )
        assert r1 == r2

        p1 = preprocessing_fingerprint({"normalize": True, "chunk_size": 512})
        p2 = preprocessing_fingerprint({"chunk_size": 512, "normalize": True})
        assert p1 == p2

    def test_manifest_keys_change_when_fingerprint_changes(self):
        dataset_fp = dataset_fingerprint("pubmed")
        model_fp = model_fingerprint("labse")
        retrieval_fp = retrieval_fingerprint(
            vector_store_type="faiss",
            retrieval_strategy={"core": {"mode": "dense"}},
        )
        preprocessing_fp = preprocessing_fingerprint({"normalize": True})

        key1 = vector_db_manifest_key(
            dataset_fp=dataset_fp,
            model_fp=model_fp,
            retrieval_fp=retrieval_fp,
            preprocessing_fp=preprocessing_fp,
        )
        key2 = vector_db_manifest_key(
            dataset_fp=dataset_fp,
            model_fp=model_fp,
            retrieval_fp=retrieval_fp,
            preprocessing_fp=preprocessing_fingerprint({"normalize": False}),
        )
        assert key1 != key2

        unique1 = unique_texts_manifest_key(
            dataset_fp=dataset_fp,
            preprocessing_fp=preprocessing_fp,
        )
        unique2 = unique_texts_manifest_key(
            dataset_fp=dataset_fp,
            preprocessing_fp=preprocessing_fingerprint({"normalize": False}),
        )
        assert unique1 != unique2

    def test_default_schema_version_constant(self):
        assert isinstance(CACHE_SCHEMA_VERSION, str)
        assert CACHE_SCHEMA_VERSION
