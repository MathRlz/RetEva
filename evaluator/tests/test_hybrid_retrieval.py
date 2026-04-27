"""Tests for hybrid retrieval combining dense and sparse methods."""
import pytest
import numpy as np
from typing import List, Tuple, Any

from evaluator.models.retrieval.rag.sparse import BM25Retriever
from evaluator.models.retrieval.rag.hybrid import HybridRetriever, reciprocal_rank_fusion


class TestBM25Retriever:
    """Tests for BM25Retriever class."""
    
    def test_index_and_search(self):
        """Test basic indexing and search functionality."""
        retriever = BM25Retriever()
        texts = [
            "diabetes treatment options",
            "heart disease symptoms",
            "cancer prevention methods",
            "diabetes management tips",
        ]
        retriever.index(texts)
        
        results = retriever.search("diabetes", k=2)
        
        assert len(results) == 2
        # Diabetes-related docs should be top results
        indices = [idx for idx, _ in results]
        assert 0 in indices or 3 in indices
        
    def test_search_returns_scores(self):
        """Test that search returns valid scores."""
        retriever = BM25Retriever()
        texts = ["machine learning", "deep learning", "natural language processing"]
        retriever.index(texts)
        
        results = retriever.search("learning", k=3)
        
        for idx, score in results:
            assert isinstance(idx, int)
            assert isinstance(score, float)
            assert score >= 0.0
    
    def test_search_without_index_raises(self):
        """Test that searching without indexing raises an error."""
        retriever = BM25Retriever()
        
        with pytest.raises(RuntimeError):
            retriever.search("test query", k=5)
    
    def test_empty_query(self):
        """Test that empty query returns empty results."""
        retriever = BM25Retriever()
        retriever.index(["document one", "document two"])
        
        results = retriever.search("", k=5)
        
        assert results == []
    
    def test_doc_count(self):
        """Test document count property."""
        retriever = BM25Retriever()
        texts = ["doc1", "doc2", "doc3"]
        retriever.index(texts)
        
        assert retriever.doc_count == 3
    
    def test_get_text(self):
        """Test retrieving original text by index."""
        retriever = BM25Retriever()
        texts = ["first document", "second document"]
        retriever.index(texts)
        
        assert retriever.get_text(0) == "first document"
        assert retriever.get_text(1) == "second document"
    
    def test_custom_bm25_params(self):
        """Test BM25 with custom k1 and b parameters."""
        retriever = BM25Retriever(k1=2.0, b=0.5)
        texts = ["short", "this is a much longer document with more terms"]
        retriever.index(texts)
        
        results = retriever.search("document", k=2)
        
        assert len(results) >= 1


class TestReciprocalRankFusion:
    """Tests for RRF function."""
    
    def test_basic_rrf(self):
        """Test basic RRF combination of two rankings."""
        ranking1 = [("a", 1.0), ("b", 0.8), ("c", 0.6)]
        ranking2 = [("b", 1.0), ("c", 0.9), ("a", 0.5)]
        
        results = reciprocal_rank_fusion([ranking1, ranking2], k=60, top_n=3)
        
        assert len(results) == 3
        # Items should be sorted by RRF score
        items = [item for item, _ in results]
        assert "a" in items
        assert "b" in items
        assert "c" in items
    
    def test_rrf_with_different_k(self):
        """Test that different k values affect ranking."""
        ranking1 = [("a", 1.0), ("b", 0.5)]
        ranking2 = [("b", 1.0), ("a", 0.5)]
        
        results_k60 = reciprocal_rank_fusion([ranking1, ranking2], k=60)
        results_k1 = reciprocal_rank_fusion([ranking1, ranking2], k=1)
        
        # Both should produce results
        assert len(results_k60) == 2
        assert len(results_k1) == 2
    
    def test_rrf_with_top_n(self):
        """Test limiting results with top_n."""
        ranking = [("a", 1.0), ("b", 0.8), ("c", 0.6), ("d", 0.4)]
        
        results = reciprocal_rank_fusion([ranking], k=60, top_n=2)
        
        assert len(results) == 2
    
    def test_rrf_empty_rankings(self):
        """Test RRF with empty rankings."""
        results = reciprocal_rank_fusion([[], []], k=60)
        
        assert results == []
    
    def test_rrf_single_ranking(self):
        """Test RRF with a single ranking."""
        ranking = [("a", 1.0), ("b", 0.5)]
        
        results = reciprocal_rank_fusion([ranking], k=60)
        
        assert len(results) == 2
        assert results[0][0] == "a"


class MockDenseRetriever:
    """Mock dense retriever for testing."""
    
    def __init__(self, results: List[Tuple[Any, float]]):
        self.results = results
    
    def search(self, query_embedding: np.ndarray, k: int) -> List[Tuple[Any, float]]:
        return self.results[:k]


class MockSparseRetriever:
    """Mock sparse retriever for testing."""
    
    def __init__(self, texts: List[str], results: List[Tuple[int, float]]):
        self._texts = texts
        self.results = results
    
    def search(self, query: str, k: int) -> List[Tuple[int, float]]:
        return self.results[:k]
    
    def get_text(self, idx: int) -> str:
        return self._texts[idx]


class TestHybridRetriever:
    """Tests for HybridRetriever class."""
    
    def test_weighted_fusion(self):
        """Test hybrid search with weighted fusion."""
        texts = ["doc0", "doc1", "doc2"]
        dense_results = [
            ({"doc_id": 0, "text": "doc0"}, 1.0),
            ({"doc_id": 1, "text": "doc1"}, 0.5),
        ]
        sparse_results = [(1, 1.0), (2, 0.5)]
        
        dense_retriever = MockDenseRetriever(dense_results)
        sparse_retriever = MockSparseRetriever(texts, sparse_results)
        
        hybrid = HybridRetriever(
            dense_retriever,
            sparse_retriever,
            dense_weight=0.5,
            fusion_method="weighted"
        )
        
        query_emb = np.random.randn(768).astype(np.float32)
        results = hybrid.search(query_emb, "test query", k=3)
        
        assert len(results) <= 3
        for payload, score in results:
            assert isinstance(score, float)
    
    def test_rrf_fusion(self):
        """Test hybrid search with RRF fusion."""
        texts = ["doc0", "doc1", "doc2"]
        dense_results = [
            ({"doc_id": 0, "text": "doc0"}, 1.0),
            ({"doc_id": 1, "text": "doc1"}, 0.5),
        ]
        sparse_results = [(1, 1.0), (0, 0.8)]
        
        dense_retriever = MockDenseRetriever(dense_results)
        sparse_retriever = MockSparseRetriever(texts, sparse_results)
        
        hybrid = HybridRetriever(
            dense_retriever,
            sparse_retriever,
            fusion_method="rrf",
            rrf_k=60
        )
        
        query_emb = np.random.randn(768).astype(np.float32)
        results = hybrid.search(query_emb, "test query", k=2)
        
        assert len(results) == 2
    
    def test_search_batch(self):
        """Test batch hybrid search."""
        texts = ["doc0", "doc1"]
        dense_results = [({"doc_id": 0, "text": "doc0"}, 1.0)]
        sparse_results = [(0, 1.0)]
        
        dense_retriever = MockDenseRetriever(dense_results)
        sparse_retriever = MockSparseRetriever(texts, sparse_results)
        
        hybrid = HybridRetriever(dense_retriever, sparse_retriever)
        
        query_embs = np.random.randn(3, 768).astype(np.float32)
        query_texts = ["query1", "query2", "query3"]
        
        results = hybrid.search_batch(query_embs, query_texts, k=1)
        
        assert len(results) == 3
        for result_list in results:
            assert len(result_list) <= 1
    
    def test_dense_weight_effect(self):
        """Test that dense_weight affects results."""
        texts = ["doc0", "doc1"]
        # Dense prefers doc0, sparse prefers doc1
        dense_results = [
            ({"doc_id": 0, "text": "doc0"}, 1.0),
            ({"doc_id": 1, "text": "doc1"}, 0.1),
        ]
        sparse_results = [(1, 1.0), (0, 0.1)]
        
        dense_retriever = MockDenseRetriever(dense_results)
        sparse_retriever = MockSparseRetriever(texts, sparse_results)
        
        # Heavy dense weight
        hybrid_dense = HybridRetriever(dense_retriever, sparse_retriever, dense_weight=0.9)
        # Heavy sparse weight
        hybrid_sparse = HybridRetriever(dense_retriever, sparse_retriever, dense_weight=0.1)
        
        query_emb = np.random.randn(768).astype(np.float32)
        
        results_dense = hybrid_dense.search(query_emb, "query", k=2)
        results_sparse = hybrid_sparse.search(query_emb, "query", k=2)
        
        # Both should return results but potentially in different orders
        assert len(results_dense) == 2
        assert len(results_sparse) == 2


class TestRetrievalPipelineHybridIntegration:
    """Integration tests for hybrid retrieval in RetrievalPipeline."""
    
    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock vector store."""
        from evaluator.storage.vector_store import InMemoryVectorStore
        store = InMemoryVectorStore()
        return store
    
    def test_pipeline_with_rrf_fusion(self, mock_vector_store):
        """Test RetrievalPipeline with RRF fusion method."""
        from evaluator.pipeline import RetrievalPipeline
        
        pipeline = RetrievalPipeline(
            mock_vector_store,
            retrieval_mode="hybrid",
            hybrid_dense_weight=0.5,
            hybrid_fusion_method="rrf",
            rrf_k=60,
        )
        
        # Build index
        embeddings = np.random.randn(5, 768).astype(np.float32)
        metadata = [
            {"doc_id": i, "text": f"document {i} about topic"} 
            for i in range(5)
        ]
        pipeline.build_index(embeddings, metadata)
        
        # Search
        query_embs = np.random.randn(2, 768).astype(np.float32)
        query_texts = ["document topic", "another query"]
        
        results = pipeline.search_batch(query_embs, k=3, query_texts=query_texts)
        
        assert len(results) == 2
        for result_list in results:
            assert len(result_list) <= 3
    
    def test_pipeline_with_weighted_fusion(self, mock_vector_store):
        """Test RetrievalPipeline with weighted fusion method."""
        from evaluator.pipeline import RetrievalPipeline
        
        pipeline = RetrievalPipeline(
            mock_vector_store,
            retrieval_mode="hybrid",
            hybrid_dense_weight=0.6,
            hybrid_fusion_method="weighted",
        )
        
        # Build index
        embeddings = np.random.randn(5, 768).astype(np.float32)
        metadata = [{"doc_id": i, "text": f"doc {i}"} for i in range(5)]
        pipeline.build_index(embeddings, metadata)
        
        # Search
        query_embs = np.random.randn(1, 768).astype(np.float32)
        query_texts = ["doc"]
        
        results = pipeline.search_batch(query_embs, k=3, query_texts=query_texts)
        
        assert len(results) == 1

    def test_pipeline_with_max_score_fusion(self, mock_vector_store):
        """Test RetrievalPipeline with max_score fusion method."""
        from evaluator.pipeline import RetrievalPipeline

        pipeline = RetrievalPipeline(
            mock_vector_store,
            retrieval_mode="hybrid",
            hybrid_dense_weight=0.6,
            hybrid_fusion_method="max_score",
        )

        embeddings = np.random.randn(5, 768).astype(np.float32)
        metadata = [{"doc_id": i, "text": f"doc {i}"} for i in range(5)]
        pipeline.build_index(embeddings, metadata)

        query_embs = np.random.randn(1, 768).astype(np.float32)
        query_texts = ["doc"]
        results = pipeline.search_batch(query_embs, k=3, query_texts=query_texts)
        assert len(results) == 1
    
    def test_pipeline_sparse_only(self, mock_vector_store):
        """Test RetrievalPipeline with sparse-only mode."""
        from evaluator.pipeline import RetrievalPipeline
        
        pipeline = RetrievalPipeline(
            mock_vector_store,
            retrieval_mode="sparse",
        )
        
        # Build index
        embeddings = np.random.randn(3, 768).astype(np.float32)
        metadata = [
            {"text": "diabetes treatment"},
            {"text": "heart disease"},
            {"text": "diabetes symptoms"},
        ]
        pipeline.build_index(embeddings, metadata)
        
        # Search - sparse doesn't use embeddings, just texts
        query_embs = np.random.randn(1, 768).astype(np.float32)
        query_texts = ["diabetes"]
        
        results = pipeline.search_batch(query_embs, k=2, query_texts=query_texts)
        
        assert len(results) == 1
        assert len(results[0]) <= 2


class TestConfigIntegration:
    """Test config integration for hybrid retrieval."""
    
    def test_vectordb_config_has_hybrid_fields(self):
        """Test that VectorDBConfig has all hybrid retrieval fields."""
        from evaluator.config import VectorDBConfig
        
        config = VectorDBConfig(
            retrieval_mode="hybrid",
            hybrid_dense_weight=0.6,
            hybrid_fusion_method="rrf",
            rrf_k=60,
            bm25_k1=1.5,
            bm25_b=0.75,
        )
        
        assert config.retrieval_mode == "hybrid"
        assert config.hybrid_dense_weight == 0.6
        assert config.hybrid_fusion_method == "rrf"
        assert config.rrf_k == 60
        assert config.bm25_k1 == 1.5
        assert config.bm25_b == 0.75
    
    def test_default_fusion_method_is_weighted(self):
        """Test that default fusion method is weighted."""
        from evaluator.config import VectorDBConfig
        
        config = VectorDBConfig()
        
        assert config.hybrid_fusion_method == "weighted"
        assert config.rrf_k == 60

    def test_invalid_hybrid_fusion_method_raises(self):
        """Invalid fusion strategy should fail fast in config."""
        from evaluator.config import VectorDBConfig

        with pytest.raises(ValueError, match="hybrid_fusion_method must be one of"):
            VectorDBConfig(hybrid_fusion_method="invalid-fusion")
