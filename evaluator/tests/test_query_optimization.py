"""Tests for query optimization module."""

import pytest
from unittest.mock import Mock, patch
from evaluator.config import QueryOptimizationConfig
from evaluator.models.retrieval.query.optimization import (
    rewrite_query,
    generate_hypothetical_document,
    decompose_query,
    generate_multi_queries,
    combine_retrieval_results,
    clear_llm_cache,
    _call_llm,
    _hash_request,
)


@pytest.fixture
def query_config():
    """Create a query optimization config for testing."""
    return QueryOptimizationConfig(
        enabled=True,
        method="rewrite",
        llm_model="gpt-4o-mini",
        llm_temperature=0.3,
        max_iterations=2,
    )


@pytest.fixture
def mock_llm_response():
    """Mock LLM API response."""
    def _mock_response(content):
        return Mock(
            status_code=200,
            json=lambda: {
                "choices": [
                    {"message": {"content": content}}
                ]
            }
        )
    return _mock_response


class TestLLMCalling:
    """Tests for LLM API calling and caching."""
    
    def test_hash_request(self):
        """Test request hashing for cache keys."""
        hash1 = _hash_request("sys1", "user1", "model1", 0.5)
        hash2 = _hash_request("sys1", "user1", "model1", 0.5)
        hash3 = _hash_request("sys1", "user2", "model1", 0.5)
        
        assert hash1 == hash2
        assert hash1 != hash3
    
    @patch('evaluator.models.retrieval.query.optimization.requests.post')
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_call_llm_success(self, mock_post, query_config, mock_llm_response):
        """Test successful LLM API call."""
        clear_llm_cache()
        mock_post.return_value = mock_llm_response("refined query")
        
        result = _call_llm("system", "user", query_config)
        
        assert result == "refined query"
        assert mock_post.called
        
        # Check request payload
        call_args = mock_post.call_args
        payload = call_args[1]['json']
        assert payload['model'] == 'gpt-4o-mini'
        assert payload['temperature'] == 0.3
        assert len(payload['messages']) == 2
    
    @patch('evaluator.models.retrieval.query.optimization.requests.post')
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_call_llm_caching(self, mock_post, query_config, mock_llm_response):
        """Test LLM response caching."""
        clear_llm_cache()
        mock_post.return_value = mock_llm_response("cached response")
        
        # First call
        result1 = _call_llm("system", "user", query_config)
        assert result1 == "cached response"
        assert mock_post.call_count == 1
        
        # Second call should use cache
        result2 = _call_llm("system", "user", query_config)
        assert result2 == "cached response"
        assert mock_post.call_count == 1  # No additional call
    
    @patch.dict('os.environ', {}, clear=True)
    def test_call_llm_missing_api_key(self, query_config):
        """Test error when API key is missing."""
        clear_llm_cache()
        
        with pytest.raises(RuntimeError, match="API key not found"):
            _call_llm("system", "user", query_config)


class TestQueryRewriting:
    """Tests for query rewriting."""
    
    @patch('evaluator.models.retrieval.query.optimization._call_llm')
    def test_rewrite_query_basic(self, mock_llm, query_config):
        """Test basic query rewriting."""
        mock_llm.return_value = "hypertension management guidelines"
        
        result = rewrite_query("htn treatment", query_config)
        
        assert result == "hypertension management guidelines"
        assert mock_llm.called
    
    @patch('evaluator.models.retrieval.query.optimization._call_llm')
    def test_rewrite_query_iterations(self, mock_llm, query_config):
        """Test iterative query refinement."""
        query_config.max_iterations = 3
        
        # Mock progressive refinement
        mock_llm.side_effect = [
            "hypertension treatment",
            "hypertension pharmacological management",
            "hypertension pharmacological management guidelines",
        ]
        
        result = rewrite_query("htn treatment", query_config)
        
        assert mock_llm.call_count == 3
        assert "guidelines" in result
    
    @patch('evaluator.models.retrieval.query.optimization._call_llm')
    def test_rewrite_query_with_context(self, mock_llm, query_config):
        """Test query rewriting with retrieval context."""
        query_config.use_initial_context = True
        query_config.context_top_k = 2
        
        context = [
            "Document about hypertension medications",
            "Clinical guidelines for blood pressure management",
        ]
        
        mock_llm.return_value = "antihypertensive medication guidelines"
        
        result = rewrite_query("htn treatment", query_config, context=context)
        
        assert result == "antihypertensive medication guidelines"
        # Second iteration should include context
        assert mock_llm.call_count >= 1
    
    @patch('evaluator.models.retrieval.query.optimization._call_llm')
    def test_rewrite_query_convergence(self, mock_llm, query_config):
        """Test early stopping when query converges."""
        query_config.max_iterations = 5
        
        # Return same query twice
        mock_llm.side_effect = [
            "refined query",
            "refined query",
        ]
        
        result = rewrite_query("original query", query_config)
        
        assert result == "refined query"
        assert mock_llm.call_count == 2  # Stopped early
    
    def test_rewrite_query_disabled(self, query_config):
        """Test that rewriting is skipped when disabled."""
        query_config.enabled = False
        
        result = rewrite_query("original query", query_config)
        
        assert result == "original query"
    
    @patch('evaluator.models.retrieval.query.optimization._call_llm')
    def test_rewrite_query_error_handling(self, mock_llm, query_config):
        """Test fallback to original query on error."""
        mock_llm.side_effect = RuntimeError("API error")
        
        result = rewrite_query("original query", query_config)
        
        assert result == "original query"


class TestHyDE:
    """Tests for HyDE (Hypothetical Document Embeddings)."""
    
    @patch('evaluator.models.retrieval.query.optimization._call_llm')
    def test_generate_hypothetical_document(self, mock_llm, query_config):
        """Test HyDE document generation."""
        query_config.method = "hyde"
        
        mock_llm.return_value = (
            "Diabetes mellitus is a metabolic disorder characterized by "
            "hyperglycemia resulting from defects in insulin secretion."
        )
        
        result = generate_hypothetical_document("What causes diabetes?", query_config)
        
        assert "Diabetes mellitus" in result
        assert "insulin" in result
        assert mock_llm.called
    
    def test_hyde_disabled(self, query_config):
        """Test that HyDE is skipped when disabled."""
        query_config.method = "rewrite"
        
        result = generate_hypothetical_document("query", query_config)
        
        assert result == "query"
    
    @patch('evaluator.models.retrieval.query.optimization._call_llm')
    def test_hyde_error_handling(self, mock_llm, query_config):
        """Test fallback to original query on error."""
        query_config.method = "hyde"
        mock_llm.side_effect = RuntimeError("API error")
        
        result = generate_hypothetical_document("query", query_config)
        
        assert result == "query"


class TestQueryDecomposition:
    """Tests for query decomposition."""
    
    @patch('evaluator.models.retrieval.query.optimization._call_llm')
    def test_decompose_query_basic(self, mock_llm, query_config):
        """Test basic query decomposition."""
        query_config.method = "decompose"
        
        mock_llm.return_value = (
            "diabetes symptoms\n"
            "diabetes treatment options\n"
            "diabetes complications"
        )
        
        result = decompose_query(
            "What are the symptoms, treatments, and complications of diabetes?",
            query_config
        )
        
        assert len(result) == 3
        assert "symptoms" in result[0]
        assert "treatment" in result[1]
        assert "complications" in result[2]
    
    @patch('evaluator.models.retrieval.query.optimization._call_llm')
    def test_decompose_query_with_numbering(self, mock_llm, query_config):
        """Test decomposition with numbered output."""
        query_config.method = "decompose"
        
        mock_llm.return_value = (
            "1. diabetes symptoms\n"
            "2. diabetes risk factors\n"
            "3. diabetes prevention strategies"
        )
        
        result = decompose_query("diabetes info", query_config)
        
        assert len(result) == 3
        assert "1." not in result[0]  # Numbering removed
        assert "symptoms" in result[0]
    
    def test_decompose_query_disabled(self, query_config):
        """Test that decomposition is skipped when disabled."""
        query_config.method = "rewrite"
        
        result = decompose_query("complex query", query_config)
        
        assert result == ["complex query"]
    
    @patch('evaluator.models.retrieval.query.optimization._call_llm')
    def test_decompose_query_empty_response(self, mock_llm, query_config):
        """Test handling of empty LLM response."""
        query_config.method = "decompose"
        mock_llm.return_value = ""
        
        result = decompose_query("query", query_config)
        
        assert result == ["query"]


class TestMultiQuery:
    """Tests for multi-query generation."""
    
    @patch('evaluator.models.retrieval.query.optimization._call_llm')
    def test_generate_multi_queries_basic(self, mock_llm, query_config):
        """Test basic multi-query generation."""
        query_config.method = "multi_query"
        
        mock_llm.return_value = (
            "myocardial infarction symptoms\n"
            "heart attack clinical presentation\n"
            "acute coronary syndrome signs"
        )
        
        result = generate_multi_queries("heart attack symptoms", query_config)
        
        assert len(result) == 4  # Original + 3 variations
        assert result[0] == "heart attack symptoms"
        assert "myocardial infarction" in result[1]
    
    @patch('evaluator.models.retrieval.query.optimization._call_llm')
    def test_generate_multi_queries_deduplication(self, mock_llm, query_config):
        """Test deduplication of query variations."""
        query_config.method = "multi_query"
        
        # Include duplicate of original
        mock_llm.return_value = (
            "heart attack symptoms\n"  # Duplicate
            "myocardial infarction signs\n"
            "cardiac event presentation"
        )
        
        result = generate_multi_queries("heart attack symptoms", query_config)
        
        # Should deduplicate case-insensitive
        assert len(result) == 3
    
    def test_generate_multi_queries_disabled(self, query_config):
        """Test that multi-query is skipped when disabled."""
        query_config.method = "rewrite"
        
        result = generate_multi_queries("query", query_config)
        
        assert result == ["query"]


class TestCombineResults:
    """Tests for combining retrieval results."""
    
    def test_combine_results_rrf(self):
        """Test RRF combination strategy."""
        results1 = [("doc1", 0.9), ("doc2", 0.7), ("doc3", 0.5)]
        results2 = [("doc2", 0.8), ("doc3", 0.6), ("doc4", 0.4)]
        
        combined = combine_retrieval_results(
            [results1, results2],
            strategy="rrf",
            k=3
        )
        
        assert len(combined) <= 3
        # doc2 and doc3 should rank higher (appear in both)
        doc_ids = [doc for doc, _ in combined]
        assert "doc2" in doc_ids or "doc3" in doc_ids
    
    def test_combine_results_weighted(self):
        """Test weighted combination strategy."""
        results1 = [("doc1", 0.9), ("doc2", 0.7)]
        results2 = [("doc2", 0.8), ("doc3", 0.6)]
        
        combined = combine_retrieval_results(
            [results1, results2],
            strategy="weighted",
            k=3,
            weights=[0.7, 0.3]
        )
        
        assert len(combined) <= 3
        # Check that scores are combined
        assert all(isinstance(score, (int, float)) for _, score in combined)
    
    def test_combine_results_union(self):
        """Test union combination strategy."""
        results1 = [("doc1", 0.9), ("doc2", 0.7)]
        results2 = [("doc3", 0.8), ("doc4", 0.6)]
        
        combined = combine_retrieval_results(
            [results1, results2],
            strategy="union",
            k=5
        )
        
        # Should include all unique documents
        doc_ids = [doc for doc, _ in combined]
        assert len(set(doc_ids)) == min(4, 5)
    
    def test_combine_results_intersection(self):
        """Test intersection combination strategy."""
        results1 = [("doc1", 0.9), ("doc2", 0.7), ("doc3", 0.5)]
        results2 = [("doc2", 0.8), ("doc3", 0.6), ("doc4", 0.4)]
        
        combined = combine_retrieval_results(
            [results1, results2],
            strategy="intersection",
            k=5
        )
        
        # Should only include documents in both lists
        doc_ids = [doc for doc, _ in combined]
        assert "doc2" in doc_ids
        assert "doc3" in doc_ids
        assert "doc1" not in doc_ids
        assert "doc4" not in doc_ids
    
    def test_combine_results_empty(self):
        """Test combining empty results."""
        combined = combine_retrieval_results([], strategy="rrf", k=5)
        assert combined == []
    
    def test_combine_results_single(self):
        """Test combining single result list."""
        results = [("doc1", 0.9), ("doc2", 0.7), ("doc3", 0.5)]
        
        combined = combine_retrieval_results([results], strategy="rrf", k=2)
        
        assert len(combined) == 2
        assert combined == results[:2]
    
    def test_combine_results_invalid_strategy(self):
        """Test error on invalid strategy."""
        results1 = [("doc1", 0.9)]
        results2 = [("doc2", 0.8)]
        
        with pytest.raises(ValueError, match="Unknown combination strategy"):
            combine_retrieval_results(
                [results1, results2],
                strategy="invalid",
                k=5
            )


class TestCacheManagement:
    """Tests for LLM cache management."""
    
    def test_clear_cache(self):
        """Test cache clearing."""
        from evaluator.models.retrieval.query.optimization import _llm_cache
        
        # Add something to cache
        _llm_cache["test_key"] = "test_value"
        assert len(_llm_cache) > 0
        
        # Clear cache
        clear_llm_cache()
        assert len(_llm_cache) == 0


class TestConfigValidation:
    """Tests for QueryOptimizationConfig validation."""
    
    def test_valid_config(self):
        """Test creating valid config."""
        config = QueryOptimizationConfig(
            enabled=True,
            method="rewrite",
            llm_temperature=0.5,
            max_iterations=3,
        )
        
        assert config.enabled is True
        assert config.method == "rewrite"
    
    def test_invalid_method(self):
        """Test error on invalid method."""
        with pytest.raises(ValueError, match="method must be one of"):
            QueryOptimizationConfig(method="invalid")
    
    def test_invalid_strategy(self):
        """Test error on invalid combine strategy."""
        with pytest.raises(ValueError, match="combine_strategy must be one of"):
            QueryOptimizationConfig(combine_strategy="invalid")
    
    def test_invalid_temperature(self):
        """Test error on invalid temperature."""
        with pytest.raises(ValueError, match="llm_temperature must be in"):
            QueryOptimizationConfig(llm_temperature=3.0)
    
    def test_invalid_iterations(self):
        """Test error on invalid max_iterations."""
        with pytest.raises(ValueError, match="max_iterations must be >= 1"):
            QueryOptimizationConfig(max_iterations=0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
