"""Tests for advanced retrieval strategies (MMR, threshold filtering, distance metrics)."""
import pytest
import numpy as np
from typing import List, Tuple, Any

from evaluator.models.retrieval.rag.strategies import (
    DistanceMetric,
    compute_similarity,
    compute_similarity_batch,
    mmr_search,
    mmr_rerank,
    threshold_filter,
    threshold_filter_with_fallback,
)


class TestDistanceMetrics:
    """Tests for distance metric computation."""
    
    def test_cosine_similarity_identical(self):
        """Test cosine similarity of identical vectors."""
        vec = np.array([1.0, 0.0, 0.0])
        sim = compute_similarity(vec, vec, DistanceMetric.COSINE)
        assert np.isclose(sim, 1.0)
    
    def test_cosine_similarity_orthogonal(self):
        """Test cosine similarity of orthogonal vectors."""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        sim = compute_similarity(a, b, DistanceMetric.COSINE)
        assert np.isclose(sim, 0.0)
    
    def test_cosine_similarity_opposite(self):
        """Test cosine similarity of opposite vectors."""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([-1.0, 0.0, 0.0])
        sim = compute_similarity(a, b, DistanceMetric.COSINE)
        assert np.isclose(sim, -1.0)
    
    def test_dot_product_similarity(self):
        """Test dot product similarity."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        sim = compute_similarity(a, b, DistanceMetric.DOT_PRODUCT)
        expected = 1*4 + 2*5 + 3*6  # 32
        assert np.isclose(sim, expected)
    
    def test_euclidean_similarity_identical(self):
        """Test euclidean similarity of identical vectors."""
        vec = np.array([1.0, 2.0, 3.0])
        sim = compute_similarity(vec, vec, DistanceMetric.EUCLIDEAN)
        assert np.isclose(sim, 1.0)  # dist=0, sim=1/(1+0)=1
    
    def test_euclidean_similarity_distant(self):
        """Test euclidean similarity of distant vectors."""
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([3.0, 4.0, 0.0])  # distance = 5
        sim = compute_similarity(a, b, DistanceMetric.EUCLIDEAN)
        expected = 1.0 / (1.0 + 5.0)
        assert np.isclose(sim, expected)
    
    def test_unknown_metric_raises(self):
        """Test that unknown metric raises ValueError."""
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        with pytest.raises(ValueError, match="Unknown metric"):
            compute_similarity(a, b, "invalid")


class TestComputeSimilarityBatch:
    """Tests for batch similarity computation."""
    
    def test_batch_cosine_similarity(self):
        """Test batch cosine similarity computation."""
        query = np.array([1.0, 0.0, 0.0])
        candidates = np.array([
            [1.0, 0.0, 0.0],  # identical
            [0.0, 1.0, 0.0],  # orthogonal
            [-1.0, 0.0, 0.0],  # opposite
        ])
        
        sims = compute_similarity_batch(query, candidates, DistanceMetric.COSINE)
        
        assert len(sims) == 3
        assert np.isclose(sims[0], 1.0)
        assert np.isclose(sims[1], 0.0)
        assert np.isclose(sims[2], -1.0)
    
    def test_batch_dot_product(self):
        """Test batch dot product computation."""
        query = np.array([1.0, 1.0])
        candidates = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ])
        
        sims = compute_similarity_batch(query, candidates, DistanceMetric.DOT_PRODUCT)
        
        assert np.isclose(sims[0], 1.0)
        assert np.isclose(sims[1], 1.0)
        assert np.isclose(sims[2], 2.0)
    
    def test_batch_single_candidate(self):
        """Test batch similarity with single candidate."""
        query = np.array([1.0, 0.0])
        candidates = np.array([1.0, 0.0])  # 1D array
        
        sims = compute_similarity_batch(query, candidates, DistanceMetric.COSINE)
        
        assert len(sims) == 1
        assert np.isclose(sims[0], 1.0)


class TestMMRSearch:
    """Tests for Maximal Marginal Relevance search."""
    
    def test_mmr_basic(self):
        """Test basic MMR functionality."""
        query = np.array([1.0, 0.0, 0.0])
        doc_embs = np.array([
            [1.0, 0.0, 0.0],   # Most similar to query
            [0.9, 0.1, 0.0],   # Very similar to doc 0
            [0.0, 1.0, 0.0],   # Orthogonal, more diverse
        ])
        
        results = mmr_search(query, doc_embs, k=2, lambda_param=0.5)
        
        assert len(results) == 2
        # First should be most relevant
        assert results[0][0] == 0
    
    def test_mmr_high_lambda_prefers_relevance(self):
        """Test that high lambda prioritizes relevance."""
        query = np.array([1.0, 0.0, 0.0])
        doc_embs = np.array([
            [1.0, 0.0, 0.0],
            [0.95, 0.05, 0.0],
            [0.5, 0.5, 0.0],
        ])
        
        # High lambda = more relevance
        results = mmr_search(query, doc_embs, k=3, lambda_param=1.0)
        
        # Should be ordered by relevance only
        indices = [idx for idx, _ in results]
        assert indices[0] == 0  # Most similar
    
    def test_mmr_low_lambda_prefers_diversity(self):
        """Test that low lambda prioritizes diversity."""
        query = np.array([1.0, 0.0, 0.0])
        doc_embs = np.array([
            [1.0, 0.0, 0.0],   # idx 0: most similar
            [0.99, 0.01, 0.0],  # idx 1: almost identical to 0
            [0.0, 1.0, 0.0],   # idx 2: very different
        ])
        
        # Low lambda = more diversity
        results = mmr_search(query, doc_embs, k=2, lambda_param=0.1)
        
        indices = [idx for idx, _ in results]
        # Second result should prefer diversity (idx 2 over idx 1)
        assert 2 in indices
    
    def test_mmr_empty_docs(self):
        """Test MMR with empty document set."""
        query = np.array([1.0, 0.0])
        doc_embs = np.array([]).reshape(0, 2)
        
        results = mmr_search(query, doc_embs, k=5)
        
        assert results == []
    
    def test_mmr_k_larger_than_docs(self):
        """Test MMR when k > number of documents."""
        query = np.array([1.0, 0.0])
        doc_embs = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
        ])
        
        results = mmr_search(query, doc_embs, k=10)
        
        assert len(results) == 2
    
    def test_mmr_with_precomputed_scores(self):
        """Test MMR with pre-computed relevance scores."""
        query = np.array([1.0, 0.0])
        doc_embs = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.5],
        ])
        initial_scores = np.array([0.9, 0.1, 0.5])
        
        results = mmr_search(
            query, doc_embs, k=3,
            lambda_param=1.0,
            initial_scores=initial_scores
        )
        
        # With lambda=1 (pure relevance), should follow initial_scores order
        indices = [idx for idx, _ in results]
        assert indices[0] == 0


class TestMMRRerank:
    """Tests for MMR reranking of existing results."""
    
    def test_mmr_rerank_basic(self):
        """Test basic MMR reranking."""
        query = np.array([1.0, 0.0, 0.0])
        results = [
            ({"doc_id": 0}, 1.0),
            ({"doc_id": 1}, 0.9),
            ({"doc_id": 2}, 0.8),
        ]
        doc_embs = np.array([
            [1.0, 0.0, 0.0],
            [0.99, 0.01, 0.0],
            [0.0, 1.0, 0.0],
        ])
        
        reranked = mmr_rerank(query, results, doc_embs, k=2, lambda_param=0.5)
        
        assert len(reranked) == 2
        # First should still be the most relevant
        assert reranked[0][0]["doc_id"] == 0
    
    def test_mmr_rerank_empty(self):
        """Test MMR reranking with empty results."""
        query = np.array([1.0, 0.0])
        results = []
        doc_embs = np.array([]).reshape(0, 2)
        
        reranked = mmr_rerank(query, results, doc_embs, k=5)
        
        assert reranked == []


class TestThresholdFilter:
    """Tests for similarity threshold filtering."""
    
    def test_threshold_filter_basic(self):
        """Test basic threshold filtering."""
        results = [
            ("doc1", 0.9),
            ("doc2", 0.7),
            ("doc3", 0.4),
            ("doc4", 0.2),
        ]
        
        filtered = threshold_filter(results, min_score=0.5)
        
        assert len(filtered) == 2
        assert filtered[0] == ("doc1", 0.9)
        assert filtered[1] == ("doc2", 0.7)
    
    def test_threshold_filter_all_pass(self):
        """Test threshold when all results pass."""
        results = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
        
        filtered = threshold_filter(results, min_score=0.5)
        
        assert filtered == results
    
    def test_threshold_filter_none_pass(self):
        """Test threshold when no results pass."""
        results = [("a", 0.3), ("b", 0.2), ("c", 0.1)]
        
        filtered = threshold_filter(results, min_score=0.5)
        
        assert filtered == []
    
    def test_threshold_filter_empty(self):
        """Test threshold with empty results."""
        filtered = threshold_filter([], min_score=0.5)
        
        assert filtered == []
    
    def test_threshold_filter_boundary(self):
        """Test threshold at exact boundary."""
        results = [("a", 0.5), ("b", 0.49999)]
        
        filtered = threshold_filter(results, min_score=0.5)
        
        assert len(filtered) == 1
        assert filtered[0] == ("a", 0.5)


class TestThresholdFilterWithFallback:
    """Tests for threshold filtering with minimum results fallback."""
    
    def test_fallback_not_needed(self):
        """Test when enough results pass threshold."""
        results = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
        
        filtered = threshold_filter_with_fallback(
            results, min_score=0.5, min_results=2
        )
        
        assert len(filtered) == 3
    
    def test_fallback_triggered(self):
        """Test when fallback is needed."""
        results = [("a", 0.4), ("b", 0.3), ("c", 0.2)]
        
        filtered = threshold_filter_with_fallback(
            results, min_score=0.5, min_results=2
        )
        
        # Should return top 2 despite being below threshold
        assert len(filtered) == 2
        assert filtered[0] == ("a", 0.4)
        assert filtered[1] == ("b", 0.3)
    
    def test_fallback_partial(self):
        """Test partial threshold pass with fallback."""
        results = [("a", 0.6), ("b", 0.3), ("c", 0.2)]
        
        filtered = threshold_filter_with_fallback(
            results, min_score=0.5, min_results=2
        )
        
        # Only 1 passes threshold, but min_results=2, so return top 2
        assert len(filtered) == 2
    
    def test_fallback_empty_results(self):
        """Test fallback with empty results."""
        filtered = threshold_filter_with_fallback(
            [], min_score=0.5, min_results=2
        )
        
        assert filtered == []


class TestRetrievalPipelineIntegration:
    """Integration tests for advanced strategies in RetrievalPipeline."""
    
    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock vector store."""
        from evaluator.storage.vector_store import InMemoryVectorStore
        return InMemoryVectorStore()
    
    def test_pipeline_with_mmr(self, mock_vector_store):
        """Test RetrievalPipeline with MMR enabled."""
        from evaluator.pipeline import RetrievalPipeline
        
        pipeline = RetrievalPipeline(
            mock_vector_store,
            retrieval_mode="dense",
            use_mmr=True,
            mmr_lambda=0.7,
        )
        
        # Build index with embeddings
        embeddings = np.random.randn(10, 768).astype(np.float32)
        metadata = [{"doc_id": i, "text": f"doc {i}"} for i in range(10)]
        pipeline.build_index(embeddings, metadata)
        
        # Search
        query_embs = np.random.randn(2, 768).astype(np.float32)
        results = pipeline.search_batch(query_embs, k=5)
        
        assert len(results) == 2
        for result_list in results:
            assert len(result_list) <= 5
    
    def test_pipeline_with_threshold(self, mock_vector_store):
        """Test RetrievalPipeline with similarity threshold."""
        from evaluator.pipeline import RetrievalPipeline
        
        pipeline = RetrievalPipeline(
            mock_vector_store,
            retrieval_mode="dense",
            min_similarity_threshold=0.0,  # Very low threshold to get results
        )
        
        embeddings = np.random.randn(5, 768).astype(np.float32)
        metadata = [{"doc_id": i, "text": f"doc {i}"} for i in range(5)]
        pipeline.build_index(embeddings, metadata)
        
        query_embs = np.random.randn(1, 768).astype(np.float32)
        results = pipeline.search_batch(query_embs, k=3)
        
        assert len(results) == 1
    
    def test_pipeline_with_mmr_and_threshold(self, mock_vector_store):
        """Test RetrievalPipeline with both MMR and threshold."""
        from evaluator.pipeline import RetrievalPipeline
        
        pipeline = RetrievalPipeline(
            mock_vector_store,
            retrieval_mode="dense",
            use_mmr=True,
            mmr_lambda=0.5,
            min_similarity_threshold=0.0,
        )
        
        embeddings = np.random.randn(10, 768).astype(np.float32)
        metadata = [{"doc_id": i, "text": f"doc {i}"} for i in range(10)]
        pipeline.build_index(embeddings, metadata)
        
        query_embs = np.random.randn(1, 768).astype(np.float32)
        results = pipeline.search_batch(query_embs, k=5)
        
        assert len(results) == 1
        assert len(results[0]) <= 5
    
    def test_pipeline_distance_metrics(self, mock_vector_store):
        """Test RetrievalPipeline with different distance metrics."""
        from evaluator.pipeline import RetrievalPipeline
        
        for metric in ["cosine", "euclidean", "dot_product"]:
            pipeline = RetrievalPipeline(
                mock_vector_store,
                retrieval_mode="dense",
                use_mmr=True,
                distance_metric=metric,
            )
            
            embeddings = np.random.randn(5, 768).astype(np.float32)
            metadata = [{"doc_id": i, "text": f"doc {i}"} for i in range(5)]
            pipeline.build_index(embeddings, metadata)
            
            query_embs = np.random.randn(1, 768).astype(np.float32)
            results = pipeline.search_batch(query_embs, k=3)
            
            assert len(results) == 1
    
    def test_pipeline_invalid_distance_metric_raises(self, mock_vector_store):
        """Test that invalid distance metric raises error."""
        from evaluator.pipeline import RetrievalPipeline
        
        with pytest.raises(ValueError, match="Unknown distance metric"):
            RetrievalPipeline(
                mock_vector_store,
                distance_metric="invalid_metric",
            )


class TestConfigIntegration:
    """Test config integration for advanced retrieval strategies."""
    
    def test_vectordb_config_has_advanced_fields(self):
        """Test that VectorDBConfig has all advanced retrieval fields."""
        from evaluator.config import VectorDBConfig
        
        config = VectorDBConfig(
            use_mmr=True,
            mmr_lambda=0.7,
            min_similarity_threshold=0.5,
            distance_metric="euclidean",
        )
        
        assert config.use_mmr is True
        assert config.mmr_lambda == 0.7
        assert config.min_similarity_threshold == 0.5
        assert config.distance_metric == "euclidean"
    
    def test_vectordb_config_defaults(self):
        """Test VectorDBConfig default values for advanced fields."""
        from evaluator.config import VectorDBConfig
        
        config = VectorDBConfig()
        
        assert config.use_mmr is False
        assert config.mmr_lambda == 0.5
        assert config.min_similarity_threshold is None
        assert config.distance_metric == "cosine"
