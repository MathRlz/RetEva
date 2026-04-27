"""Tests for embedding fusion functionality."""

import pytest
import numpy as np
from evaluator.config import EmbeddingFusionConfig
from evaluator.models.retrieval.embedding_fusion import (
    fuse_embeddings,
    normalize_embeddings,
    reduce_dimensions,
    validate_fusion_config
)


class TestNormalizeEmbeddings:
    """Tests for normalize_embeddings function."""
    
    def test_normalize_2d_array(self):
        """Test normalization of 2D array."""
        emb = np.array([[3.0, 4.0], [5.0, 12.0]])
        normed = normalize_embeddings(emb)
        
        # Check norms are 1.0
        norms = np.linalg.norm(normed, axis=1)
        assert np.allclose(norms, 1.0)
        
        # Check shape unchanged
        assert normed.shape == emb.shape
    
    def test_normalize_1d_array(self):
        """Test normalization of 1D array."""
        emb = np.array([3.0, 4.0])
        normed = normalize_embeddings(emb)
        
        # Check norm is 1.0
        norm = np.linalg.norm(normed)
        assert np.isclose(norm, 1.0)
    
    def test_normalize_zero_vector(self):
        """Test normalization of zero vector (should not crash)."""
        emb = np.array([[0.0, 0.0], [1.0, 0.0]])
        normed = normalize_embeddings(emb)
        
        # Zero vector should remain zero
        assert np.allclose(normed[0], [0.0, 0.0])
        # Non-zero vector should be normalized
        assert np.allclose(normed[1], [1.0, 0.0])


class TestReduceDimensions:
    """Tests for reduce_dimensions function."""
    
    def test_pca_reduction(self):
        """Test PCA dimension reduction."""
        # Create embeddings with clear structure
        # Need more samples than target dimensions for PCA
        np.random.seed(42)
        emb = np.random.randn(600, 512)
        
        reduced, reducer = reduce_dimensions(emb, target_dim=256, method="pca")
        
        assert reduced.shape == (600, 256)
        assert reducer is not None
        assert hasattr(reducer, 'explained_variance_ratio_')
    
    def test_random_projection_reduction(self):
        """Test random projection dimension reduction."""
        np.random.seed(42)
        emb = np.random.randn(100, 512)
        
        reduced, reducer = reduce_dimensions(emb, target_dim=256, method="random_projection")
        
        assert reduced.shape == (100, 256)
        assert reducer is not None
    
    def test_reduction_with_prefitted_reducer(self):
        """Test reduction using a pre-fitted reducer."""
        np.random.seed(42)
        # Need more samples than target dimensions
        emb_train = np.random.randn(300, 512)
        emb_test = np.random.randn(50, 512)
        
        # Fit reducer on training data
        _, reducer = reduce_dimensions(emb_train, target_dim=256, method="pca")
        
        # Apply to test data
        reduced_test, _ = reduce_dimensions(emb_test, target_dim=256, method="pca", reducer=reducer)
        
        assert reduced_test.shape == (50, 256)
    
    def test_no_reduction_when_target_dim_larger(self):
        """Test that no reduction occurs when target dim >= current dim."""
        emb = np.random.randn(100, 256)
        
        reduced, reducer = reduce_dimensions(emb, target_dim=512, method="pca")
        
        # Should return original embeddings
        assert np.allclose(reduced, emb)
        assert reducer is None
    
    def test_invalid_method_raises_error(self):
        """Test that invalid reduction method raises error."""
        emb = np.random.randn(100, 512)
        
        with pytest.raises(ValueError, match="Unsupported dimension reduction method"):
            reduce_dimensions(emb, target_dim=256, method="invalid_method")


class TestFuseEmbeddings:
    """Tests for fuse_embeddings function."""
    
    def test_weighted_fusion_same_dimensions(self):
        """Test weighted fusion with same-dimensional embeddings."""
        np.random.seed(42)
        audio_emb = np.random.randn(10, 768)
        text_emb = np.random.randn(10, 768)
        
        config = EmbeddingFusionConfig(
            enabled=True,
            fusion_method="weighted",
            audio_weight=0.6,
            text_weight=0.4,
            normalize_before_fusion=False
        )
        
        fused, reducer = fuse_embeddings(audio_emb, text_emb, config)
        
        assert fused.shape == (10, 768)
        assert reducer is None
        
        # Check fusion is correct weighted combination
        expected = 0.6 * audio_emb + 0.4 * text_emb
        assert np.allclose(fused, expected)
    
    def test_weighted_fusion_with_normalization(self):
        """Test weighted fusion with normalization."""
        np.random.seed(42)
        audio_emb = np.random.randn(10, 768)
        text_emb = np.random.randn(10, 768)
        
        config = EmbeddingFusionConfig(
            enabled=True,
            fusion_method="weighted",
            audio_weight=0.6,
            text_weight=0.4,
            normalize_before_fusion=True
        )
        
        fused, _ = fuse_embeddings(audio_emb, text_emb, config)
        
        # Check all embeddings are normalized
        norms = np.linalg.norm(fused, axis=1)
        assert np.allclose(norms, 1.0)
    
    def test_average_fusion(self):
        """Test average fusion."""
        np.random.seed(42)
        audio_emb = np.random.randn(10, 768)
        text_emb = np.random.randn(10, 768)
        
        config = EmbeddingFusionConfig(
            enabled=True,
            fusion_method="average",
            normalize_before_fusion=False
        )
        
        fused, _ = fuse_embeddings(audio_emb, text_emb, config)
        
        expected = (audio_emb + text_emb) / 2.0
        assert np.allclose(fused, expected)
    
    def test_max_pool_fusion(self):
        """Test max pool fusion."""
        np.random.seed(42)
        audio_emb = np.random.randn(10, 768)
        text_emb = np.random.randn(10, 768)
        
        config = EmbeddingFusionConfig(
            enabled=True,
            fusion_method="max_pool",
            normalize_before_fusion=False
        )
        
        fused, _ = fuse_embeddings(audio_emb, text_emb, config)
        
        expected = np.maximum(audio_emb, text_emb)
        assert np.allclose(fused, expected)
    
    def test_concatenate_fusion(self):
        """Test concatenate fusion."""
        np.random.seed(42)
        audio_emb = np.random.randn(10, 768)
        text_emb = np.random.randn(10, 1024)
        
        config = EmbeddingFusionConfig(
            enabled=True,
            fusion_method="concatenate",
            normalize_before_fusion=False
        )
        
        fused, reducer = fuse_embeddings(audio_emb, text_emb, config)
        
        assert fused.shape == (10, 768 + 1024)
        assert reducer is None
    
    def test_concatenate_fusion_with_reduction(self):
        """Test concatenate fusion with dimension reduction."""
        np.random.seed(42)
        # Need enough samples for PCA: n_samples >= target_dim
        # Concatenated dim is 768 + 1024 = 1792, reducing to 512
        audio_emb = np.random.randn(600, 768)
        text_emb = np.random.randn(600, 1024)
        
        config = EmbeddingFusionConfig(
            enabled=True,
            fusion_method="concatenate",
            normalize_before_fusion=False,
            dimension_reduction="pca",
            target_dim=512
        )
        
        fused, reducer = fuse_embeddings(audio_emb, text_emb, config)
        
        assert fused.shape == (600, 512)
        assert reducer is not None
    
    def test_weighted_fusion_different_dimensions_raises_error(self):
        """Test that weighted fusion with different dimensions raises error."""
        audio_emb = np.random.randn(10, 768)
        text_emb = np.random.randn(10, 1024)
        
        config = EmbeddingFusionConfig(
            enabled=True,
            fusion_method="weighted"
        )
        
        with pytest.raises(ValueError, match="Weighted fusion requires same dimensions"):
            fuse_embeddings(audio_emb, text_emb, config)
    
    def test_batch_size_mismatch_raises_error(self):
        """Test that mismatched batch sizes raise error."""
        audio_emb = np.random.randn(10, 768)
        text_emb = np.random.randn(5, 768)
        
        config = EmbeddingFusionConfig(
            enabled=True,
            fusion_method="weighted"
        )
        
        with pytest.raises(ValueError, match="Batch size mismatch"):
            fuse_embeddings(audio_emb, text_emb, config)
    
    def test_1d_embeddings_converted_to_2d(self):
        """Test that 1D embeddings are converted to 2D."""
        audio_emb = np.random.randn(768)
        text_emb = np.random.randn(768)
        
        config = EmbeddingFusionConfig(
            enabled=True,
            fusion_method="weighted",
            audio_weight=0.5,
            text_weight=0.5,
            normalize_before_fusion=False
        )
        
        fused, _ = fuse_embeddings(audio_emb, text_emb, config)
        
        assert fused.shape == (1, 768)


class TestValidateFusionConfig:
    """Tests for validate_fusion_config function."""
    
    def test_validate_weighted_fusion_same_dimensions(self):
        """Test validation passes for weighted fusion with same dimensions."""
        config = EmbeddingFusionConfig(
            enabled=True,
            fusion_method="weighted"
        )
        
        # Should not raise
        validate_fusion_config(config, audio_dim=768, text_dim=768)
    
    def test_validate_weighted_fusion_different_dimensions_raises_error(self):
        """Test validation fails for weighted fusion with different dimensions."""
        config = EmbeddingFusionConfig(
            enabled=True,
            fusion_method="weighted"
        )
        
        with pytest.raises(ValueError, match="requires audio and text embeddings to have the same dimension"):
            validate_fusion_config(config, audio_dim=768, text_dim=1024)
    
    def test_validate_concatenate_with_reduction(self):
        """Test validation passes for concatenate with reduction."""
        config = EmbeddingFusionConfig(
            enabled=True,
            fusion_method="concatenate",
            dimension_reduction="pca",
            target_dim=1024
        )
        
        # Should not raise
        validate_fusion_config(config, audio_dim=768, text_dim=1024)
    
    def test_validate_dimension_reduction_without_concatenate_raises_error(self):
        """Test validation fails when reduction is enabled without concatenate."""
        # This error is caught in __post_init__, so we test it with pytest.raises
        with pytest.raises(ValueError, match="dimension_reduction only applies when fusion_method='concatenate'"):
            EmbeddingFusionConfig(
                enabled=True,
                fusion_method="weighted",
                dimension_reduction="pca",
                target_dim=512
            )
    
    def test_validate_disabled_config_passes(self):
        """Test validation passes for disabled config."""
        config = EmbeddingFusionConfig(enabled=False)
        
        # Should not raise even with incompatible dimensions
        validate_fusion_config(config, audio_dim=768, text_dim=1024)


class TestEmbeddingFusionConfig:
    """Tests for EmbeddingFusionConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = EmbeddingFusionConfig()
        
        assert config.enabled is False
        assert config.audio_weight == 0.5
        assert config.text_weight == 0.5
        assert config.fusion_method == "weighted"
        assert config.normalize_before_fusion is True
        assert config.dimension_reduction is None
        assert config.target_dim is None
        assert config.require_same_dimensions is False
    
    def test_invalid_audio_weight_raises_error(self):
        """Test that invalid audio weight raises error."""
        with pytest.raises(ValueError, match="audio_weight must be in"):
            EmbeddingFusionConfig(audio_weight=1.5)
    
    def test_invalid_text_weight_raises_error(self):
        """Test that invalid text weight raises error."""
        with pytest.raises(ValueError, match="text_weight must be in"):
            EmbeddingFusionConfig(text_weight=-0.1)
    
    def test_invalid_fusion_method_raises_error(self):
        """Test that invalid fusion method raises error."""
        with pytest.raises(ValueError, match="fusion_method must be one of"):
            EmbeddingFusionConfig(fusion_method="invalid")
    
    def test_dimension_reduction_without_concatenate_raises_error(self):
        """Test that dimension reduction without concatenate raises error."""
        with pytest.raises(ValueError, match="dimension_reduction only applies when fusion_method='concatenate'"):
            EmbeddingFusionConfig(
                fusion_method="weighted",
                dimension_reduction="pca",
                target_dim=512
            )
    
    def test_dimension_reduction_without_target_dim_raises_error(self):
        """Test that dimension reduction without target_dim raises error."""
        with pytest.raises(ValueError, match="target_dim must be specified"):
            EmbeddingFusionConfig(
                fusion_method="concatenate",
                dimension_reduction="pca",
                target_dim=None
            )
    
    def test_valid_concatenate_with_reduction(self):
        """Test valid concatenate configuration with reduction."""
        config = EmbeddingFusionConfig(
            enabled=True,
            fusion_method="concatenate",
            dimension_reduction="pca",
            target_dim=1024
        )
        
        assert config.fusion_method == "concatenate"
        assert config.dimension_reduction == "pca"
        assert config.target_dim == 1024


class TestIntegrationFusion:
    """Integration tests for embedding fusion."""
    
    def test_full_fusion_pipeline_weighted(self):
        """Test complete fusion pipeline with weighted method."""
        np.random.seed(42)
        audio_emb = np.random.randn(20, 768)
        text_emb = np.random.randn(20, 768)
        
        config = EmbeddingFusionConfig(
            enabled=True,
            fusion_method="weighted",
            audio_weight=0.7,
            text_weight=0.3,
            normalize_before_fusion=True
        )
        
        # Validate config
        validate_fusion_config(config, audio_dim=768, text_dim=768)
        
        # Perform fusion
        fused, _ = fuse_embeddings(audio_emb, text_emb, config)
        
        assert fused.shape == (20, 768)
        # Check all embeddings are normalized
        norms = np.linalg.norm(fused, axis=1)
        assert np.allclose(norms, 1.0)
    
    def test_full_fusion_pipeline_concatenate_with_pca(self):
        """Test complete fusion pipeline with concatenate and PCA."""
        np.random.seed(42)
        # Need enough samples for PCA: n_samples >= target_dim
        # Concatenated dim is 768 + 1024 = 1792, reducing to 768
        audio_emb = np.random.randn(800, 768)
        text_emb = np.random.randn(800, 1024)
        
        config = EmbeddingFusionConfig(
            enabled=True,
            fusion_method="concatenate",
            dimension_reduction="pca",
            target_dim=768,
            normalize_before_fusion=True
        )
        
        # Validate config
        validate_fusion_config(config, audio_dim=768, text_dim=1024)
        
        # Perform fusion
        fused, reducer = fuse_embeddings(audio_emb, text_emb, config)
        
        assert fused.shape == (800, 768)
        assert reducer is not None
        
        # Check normalization
        norms = np.linalg.norm(fused, axis=1)
        assert np.allclose(norms, 1.0)
        
        # Test applying to new data
        audio_emb_new = np.random.randn(10, 768)
        text_emb_new = np.random.randn(10, 1024)
        fused_new, _ = fuse_embeddings(audio_emb_new, text_emb_new, config, reducer=reducer)
        
        assert fused_new.shape == (10, 768)
