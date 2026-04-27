"""Tests for the constants module."""

import pytest

from evaluator.constants import (
    MIN_NORM_THRESHOLD,
    DEFAULT_SAMPLE_RATE,
    LABSE_DIM,
    JINA_V4_DIM,
    BGE_M3_DIM,
    NEMOTRON_DIM,
    CLIP_DIM,
    DEFAULT_AUDIO_EMB_DIM,
    DEFAULT_BATCH_SIZE,
    DEFAULT_FAISS_NLIST,
)


class TestNumericalConstants:
    """Test numerical constants."""

    def test_min_norm_threshold_exists(self):
        """MIN_NORM_THRESHOLD should be a small positive value."""
        assert MIN_NORM_THRESHOLD > 0
        assert MIN_NORM_THRESHOLD < 1e-6

    def test_min_norm_threshold_value(self):
        """MIN_NORM_THRESHOLD should be 1e-12."""
        assert MIN_NORM_THRESHOLD == 1e-12


class TestAudioConstants:
    """Test audio-related constants."""

    def test_default_sample_rate(self):
        """DEFAULT_SAMPLE_RATE should be 16000 Hz."""
        assert DEFAULT_SAMPLE_RATE == 16000


class TestEmbeddingDimensions:
    """Test embedding dimension constants."""

    def test_labse_dim(self):
        """LaBSE embedding dimension."""
        assert LABSE_DIM == 768

    def test_jina_v4_dim(self):
        """Jina V4 embedding dimension."""
        assert JINA_V4_DIM == 1024

    def test_bge_m3_dim(self):
        """BGE-M3 embedding dimension."""
        assert BGE_M3_DIM == 1024

    def test_nemotron_dim(self):
        """Nemotron embedding dimension."""
        assert NEMOTRON_DIM == 1024

    def test_clip_dim(self):
        """CLIP embedding dimension."""
        assert CLIP_DIM == 768

    def test_default_audio_emb_dim(self):
        """Default audio embedding dimension."""
        assert DEFAULT_AUDIO_EMB_DIM == 1024


class TestBatchSizes:
    """Test batch size constants."""

    def test_default_batch_size(self):
        """Default batch size should be 32."""
        assert DEFAULT_BATCH_SIZE == 32


class TestFaissParameters:
    """Test FAISS index parameters."""

    def test_default_faiss_nlist(self):
        """Default FAISS nlist should be 1024."""
        assert DEFAULT_FAISS_NLIST == 1024


class TestConstantsExport:
    """Test that constants are properly exported from the package."""

    def test_constants_importable_from_advanced_api(self):
        """Constants should be importable from the advanced API module."""
        from evaluator.advanced_api.constants import (
            MIN_NORM_THRESHOLD,
            DEFAULT_SAMPLE_RATE,
            LABSE_DIM,
            JINA_V4_DIM,
            BGE_M3_DIM,
            NEMOTRON_DIM,
            CLIP_DIM,
            DEFAULT_AUDIO_EMB_DIM,
            DEFAULT_BATCH_SIZE,
            DEFAULT_FAISS_NLIST,
        )
        # Verify they have expected types
        assert isinstance(MIN_NORM_THRESHOLD, float)
        assert isinstance(DEFAULT_SAMPLE_RATE, int)
        assert isinstance(DEFAULT_BATCH_SIZE, int)
