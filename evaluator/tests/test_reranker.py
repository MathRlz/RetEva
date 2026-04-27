"""Tests for cross-encoder reranking functionality."""
import pytest
import numpy as np
from typing import List, Tuple, Any
from unittest.mock import MagicMock, patch

from evaluator.models.retrieval.rag.reranker import BaseReranker, CrossEncoderReranker, _extract_text


class TestExtractText:
    """Tests for _extract_text helper function."""

    def test_extract_from_dict_with_text(self):
        """Test extracting text from dict payload."""
        payload = {"text": "hello world", "doc_id": 1}
        assert _extract_text(payload) == "hello world"

    def test_extract_from_dict_without_text(self):
        """Test extracting from dict without text key."""
        payload = {"doc_id": 1}
        assert _extract_text(payload) == ""

    def test_extract_from_string(self):
        """Test extracting from string payload."""
        payload = "direct text"
        assert _extract_text(payload) == "direct text"

    def test_extract_from_number(self):
        """Test extracting from numeric payload."""
        assert _extract_text(42) == "42"


class MockCrossEncoder:
    """Mock CrossEncoder for testing without sentence-transformers dependency."""

    def __init__(self, model_name: str, max_length: int = 512, device: str = None):
        self.model_name = model_name
        self.max_length = max_length
        self.device = device

    def predict(self, pairs: List[List[str]], batch_size: int = 32) -> np.ndarray:
        """Return mock scores based on simple heuristic."""
        scores = []
        for query, doc in pairs:
            # Simple heuristic: score based on word overlap
            q_words = set(query.lower().split())
            d_words = set(doc.lower().split())
            if not q_words or not d_words:
                scores.append(0.0)
            else:
                overlap = len(q_words & d_words)
                scores.append(overlap / max(len(q_words), 1))
        return np.array(scores)


class TestCrossEncoderRerankerWithMock:
    """Tests for CrossEncoderReranker using mocked CrossEncoder."""

    @pytest.fixture
    def mock_reranker(self):
        """Create a reranker with mocked CrossEncoder."""
        with patch.dict("sys.modules", {"sentence_transformers": MagicMock()}):
            # Create a mock module structure
            mock_st = MagicMock()
            mock_st.CrossEncoder = MockCrossEncoder
            
            with patch.dict("sys.modules", {"sentence_transformers": mock_st}):
                # Need to reload or create reranker fresh
                from evaluator.models.retrieval.rag.reranker import CrossEncoderReranker
                
                # Patch the import inside the class
                with patch.object(CrossEncoderReranker, "__init__", lambda self, **kwargs: None):
                    reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
                    reranker._model_name = "mock-model"
                    reranker.batch_size = 16
                    reranker.max_length = 512
                    reranker.model = MockCrossEncoder("mock-model")
                    return reranker

    def test_init(self, mock_reranker):
        """Test reranker initialization."""
        assert mock_reranker.name() == "mock-model"
        assert mock_reranker.batch_size == 16

    def test_rerank_empty_documents(self, mock_reranker):
        """Test reranking with empty document list."""
        results = mock_reranker.rerank("test query", [])
        assert results == []

    def test_rerank_single_document(self, mock_reranker):
        """Test reranking with single document."""
        documents = [({"text": "test document", "doc_id": 0}, 0.9)]
        results = mock_reranker.rerank("test query", documents)
        
        assert len(results) == 1
        assert results[0][0]["doc_id"] == 0
        assert isinstance(results[0][1], float)

    def test_rerank_multiple_documents(self, mock_reranker):
        """Test reranking with multiple documents."""
        documents = [
            ({"text": "diabetes treatment options", "doc_id": 0}, 0.9),
            ({"text": "heart disease symptoms", "doc_id": 1}, 0.8),
            ({"text": "diabetes management tips", "doc_id": 2}, 0.7),
        ]
        results = mock_reranker.rerank("diabetes", documents)
        
        assert len(results) == 3
        # Diabetes documents should score higher
        doc_ids = [r[0]["doc_id"] for r in results]
        # Either doc 0 or 2 should be first (both contain "diabetes")
        assert doc_ids[0] in [0, 2]

    def test_rerank_with_top_k(self, mock_reranker):
        """Test reranking with top_k limit."""
        documents = [
            ({"text": "doc1", "doc_id": 0}, 0.9),
            ({"text": "doc2", "doc_id": 1}, 0.8),
            ({"text": "doc3", "doc_id": 2}, 0.7),
            ({"text": "doc4", "doc_id": 3}, 0.6),
        ]
        results = mock_reranker.rerank("test", documents, top_k=2)
        
        assert len(results) == 2

    def test_rerank_preserves_payload_structure(self, mock_reranker):
        """Test that reranking preserves payload structure."""
        documents = [
            ({"text": "test document", "doc_id": 42, "extra": "data"}, 0.9),
        ]
        results = mock_reranker.rerank("test", documents)
        
        assert results[0][0]["doc_id"] == 42
        assert results[0][0]["extra"] == "data"
        assert results[0][0]["text"] == "test document"

    def test_rerank_batch(self, mock_reranker):
        """Test batch reranking."""
        queries = ["diabetes", "heart"]
        documents_batch = [
            [
                ({"text": "diabetes info", "doc_id": 0}, 0.9),
                ({"text": "other info", "doc_id": 1}, 0.8),
            ],
            [
                ({"text": "heart health", "doc_id": 2}, 0.9),
                ({"text": "other topic", "doc_id": 3}, 0.8),
            ],
        ]
        
        results = mock_reranker.rerank_batch(queries, documents_batch)
        
        assert len(results) == 2
        assert len(results[0]) == 2
        assert len(results[1]) == 2

    def test_rerank_batch_with_top_k(self, mock_reranker):
        """Test batch reranking with top_k limit."""
        queries = ["query1", "query2"]
        documents_batch = [
            [({"text": f"doc{i}", "doc_id": i}, 0.9 - i * 0.1) for i in range(5)],
            [({"text": f"doc{i}", "doc_id": i}, 0.9 - i * 0.1) for i in range(5)],
        ]
        
        results = mock_reranker.rerank_batch(queries, documents_batch, top_k=2)
        
        assert len(results) == 2
        assert all(len(r) == 2 for r in results)

    def test_rerank_batch_mismatched_lengths(self, mock_reranker):
        """Test batch reranking with mismatched query/document counts."""
        queries = ["q1", "q2", "q3"]
        documents_batch = [
            [({"text": "doc", "doc_id": 0}, 0.9)],
            [({"text": "doc", "doc_id": 1}, 0.9)],
        ]
        
        with pytest.raises(ValueError, match="must match"):
            mock_reranker.rerank_batch(queries, documents_batch)


class TestRetrievalPipelineWithReranker:
    """Integration tests for RetrievalPipeline with cross-encoder reranker."""

    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock vector store."""
        from evaluator.storage.vector_store import InMemoryVectorStore
        return InMemoryVectorStore()

    @pytest.fixture
    def mock_reranker(self):
        """Create a mock reranker."""
        reranker = MagicMock(spec=BaseReranker)
        reranker.name.return_value = "mock-reranker"
        
        def mock_rerank(query, documents, top_k=None):
            # Reverse the order to simulate reranking effect
            reranked = [(doc, 1.0 - i * 0.1) for i, (doc, _) in enumerate(reversed(documents))]
            if top_k:
                reranked = reranked[:top_k]
            return reranked
        
        reranker.rerank.side_effect = mock_rerank
        return reranker

    def test_pipeline_with_cross_encoder_reranker(self, mock_vector_store, mock_reranker):
        """Test RetrievalPipeline with cross-encoder reranker."""
        from evaluator.pipeline import RetrievalPipeline
        
        pipeline = RetrievalPipeline(
            mock_vector_store,
            retrieval_mode="dense",
            reranker=mock_reranker,
            reranker_top_k=10,
        )
        
        # Build index
        embeddings = np.random.randn(5, 768).astype(np.float32)
        metadata = [
            {"doc_id": i, "text": f"document {i} about topic"}
            for i in range(5)
        ]
        pipeline.build_index(embeddings, metadata)
        
        # Search with reranking
        query_embs = np.random.randn(2, 768).astype(np.float32)
        query_texts = ["document topic", "another query"]
        
        results = pipeline.search_batch(query_embs, k=3, query_texts=query_texts)
        
        assert len(results) == 2
        # Verify reranker was called
        assert mock_reranker.rerank.call_count == 2

    def test_pipeline_without_reranker(self, mock_vector_store):
        """Test RetrievalPipeline without reranker."""
        from evaluator.pipeline import RetrievalPipeline
        
        pipeline = RetrievalPipeline(
            mock_vector_store,
            retrieval_mode="dense",
            reranker=None,
        )
        
        # Build index
        embeddings = np.random.randn(3, 768).astype(np.float32)
        metadata = [{"doc_id": i, "text": f"doc {i}"} for i in range(3)]
        pipeline.build_index(embeddings, metadata)
        
        # Search without reranking
        query_embs = np.random.randn(1, 768).astype(np.float32)
        
        results = pipeline.search_batch(query_embs, k=2)
        
        assert len(results) == 1
        assert len(results[0]) <= 2

    def test_pipeline_hybrid_with_reranker(self, mock_vector_store, mock_reranker):
        """Test hybrid retrieval with cross-encoder reranker."""
        from evaluator.pipeline import RetrievalPipeline
        
        pipeline = RetrievalPipeline(
            mock_vector_store,
            retrieval_mode="hybrid",
            reranker=mock_reranker,
            reranker_top_k=10,
        )
        
        # Build index
        embeddings = np.random.randn(5, 768).astype(np.float32)
        metadata = [
            {"doc_id": i, "text": f"diabetes treatment document {i}"}
            for i in range(5)
        ]
        pipeline.build_index(embeddings, metadata)
        
        # Search
        query_embs = np.random.randn(1, 768).astype(np.float32)
        query_texts = ["diabetes treatment"]
        
        results = pipeline.search_batch(query_embs, k=3, query_texts=query_texts)
        
        assert len(results) == 1
        assert mock_reranker.rerank.called


class TestConfigWithReranker:
    """Test configuration for reranker settings."""

    def test_vectordb_config_has_reranker_fields(self):
        """Test that VectorDBConfig has all reranker fields."""
        from evaluator.config import VectorDBConfig
        
        config = VectorDBConfig(
            reranker_enabled=True,
            reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            reranker_top_k=20,
            reranker_device="cuda:0",
        )
        
        assert config.reranker_enabled is True
        assert config.reranker_model == "cross-encoder/ms-marco-MiniLM-L-6-v2"
        assert config.reranker_top_k == 20
        assert config.reranker_device == "cuda:0"

    def test_vectordb_config_default_reranker_disabled(self):
        """Test that reranker is disabled by default."""
        from evaluator.config import VectorDBConfig
        
        config = VectorDBConfig()
        
        assert config.reranker_enabled is False
        assert config.reranker_model == "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def test_reranker_mode_cross_encoder(self):
        """Test reranker_mode supports cross_encoder value."""
        from evaluator.config import VectorDBConfig
        
        config = VectorDBConfig(reranker_mode="cross_encoder")
        
        assert config.reranker_mode == "cross_encoder"


class TestRerankerRegistry:
    """Test reranker model registry."""

    def test_reranker_registry_exists(self):
        """Test that reranker registry exists."""
        from evaluator.models.registry import reranker_registry
        
        assert reranker_registry is not None
        assert reranker_registry.name == "Reranker"

    def test_register_reranker_decorator(self):
        """Test register_reranker_model decorator."""
        from evaluator.models.registry import register_reranker_model, reranker_registry
        
        @register_reranker_model("test_reranker", default_name="test-model")
        class TestReranker:
            pass
        
        assert reranker_registry.is_registered("test_reranker")
        assert reranker_registry.get_default_name("test_reranker") == "test-model"


class TestRerankerFactory:
    """Test reranker factory function."""

    def test_create_reranker_registers_cross_encoder(self):
        """Test that create_reranker registers cross_encoder type."""
        from evaluator.models.factory import _register_rerankers
        from evaluator.models.registry import reranker_registry
        
        _register_rerankers()
        
        assert reranker_registry.is_registered("cross_encoder")
        assert reranker_registry.get_default_name("cross_encoder") == "cross-encoder/ms-marco-MiniLM-L-6-v2"
