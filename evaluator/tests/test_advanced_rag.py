"""Tests for advanced RAG techniques."""

import pytest
import numpy as np
from evaluator.models.retrieval.rag.advanced import (
    multi_vector_search,
    expand_query_with_synonyms,
    expand_query_with_embeddings,
    pseudo_relevance_feedback,
    adaptive_fusion_weights,
    apply_diversity_penalty,
    filter_by_similarity_thresholds,
)
from evaluator.models.retrieval.rag.strategies import (
    borda_count_fusion,
    distribution_based_fusion,
)
from evaluator.config import VectorDBConfig


class TestMultiVectorSearch:
    """Tests for multi-vector retrieval."""
    
    def test_max_sim_strategy(self):
        """Test MaxSim strategy."""
        query = np.array([1.0, 0.0, 0.0])
        
        # Doc1: two vectors, one very similar to query
        doc1_vectors = [
            np.array([0.9, 0.1, 0.0]),
            np.array([0.0, 1.0, 0.0])
        ]
        
        # Doc2: one vector, less similar
        doc2_vectors = [
            np.array([0.5, 0.5, 0.0])
        ]
        
        results = multi_vector_search(
            query,
            [doc1_vectors, doc2_vectors],
            ["doc1", "doc2"],
            strategy="max_sim",
            k=2
        )
        
        assert len(results) == 2
        # Doc1 should rank higher due to MaxSim
        assert results[0][0] == "doc1"
    
    def test_avg_sim_strategy(self):
        """Test AvgSim strategy."""
        query = np.array([1.0, 0.0, 0.0])
        
        doc1_vectors = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0])
        ]
        
        doc2_vectors = [
            np.array([0.8, 0.2, 0.0])
        ]
        
        results = multi_vector_search(
            query,
            [doc1_vectors, doc2_vectors],
            ["doc1", "doc2"],
            strategy="avg_sim",
            k=2
        )
        
        assert len(results) == 2
    
    def test_late_interaction_strategy(self):
        """Test late interaction strategy."""
        query = np.array([1.0, 0.0, 0.0])
        
        doc1_vectors = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.9, 0.1, 0.0]),
            np.array([0.8, 0.2, 0.0])
        ]
        
        results = multi_vector_search(
            query,
            [doc1_vectors],
            ["doc1"],
            strategy="late_interaction",
            k=1
        )
        
        assert len(results) == 1
        assert results[0][0] == "doc1"
    
    def test_empty_doc_vectors(self):
        """Test handling of empty document vectors."""
        query = np.array([1.0, 0.0, 0.0])
        
        results = multi_vector_search(
            query,
            [[], [np.array([1.0, 0.0, 0.0])]],
            ["doc1", "doc2"],
            strategy="max_sim",
            k=2
        )
        
        # Doc1 with no vectors should score 0
        assert results[0][0] == "doc2"


class TestQueryExpansion:
    """Tests for query expansion."""
    
    def test_expand_with_synonyms(self):
        """Test synonym-based expansion."""
        query = "htn treatment"
        expanded = expand_query_with_synonyms(query, num_terms=5)
        
        assert "htn" in expanded
        assert "hypertension" in expanded
    
    def test_expand_no_synonyms(self):
        """Test expansion when no synonyms found."""
        query = "unknown medical term xyz"
        expanded = expand_query_with_synonyms(query, num_terms=5)
        
        assert expanded == query
    
    def test_expand_with_embeddings(self):
        """Test embedding-based expansion."""
        query_emb = np.array([1.0, 0.0, 0.0])
        
        corpus_embs = np.array([
            [0.9, 0.1, 0.0],
            [0.8, 0.2, 0.0],
            [0.0, 0.0, 1.0]
        ])
        
        corpus_terms = ["hypertension", "blood pressure", "diabetes"]
        
        expanded = expand_query_with_embeddings(
            query_emb,
            corpus_embs,
            corpus_terms,
            num_terms=2
        )
        
        assert len(expanded) == 2
        assert "hypertension" in expanded or "blood pressure" in expanded


class TestPseudoRelevanceFeedback:
    """Tests for pseudo-relevance feedback."""
    
    def test_basic_feedback(self):
        """Test extracting feedback terms."""
        results = [
            ({"text": "diabetes mellitus treatment management"}, 0.9),
            ({"text": "glucose insulin therapy medication"}, 0.8),
            ({"text": "blood sugar control monitoring"}, 0.7),
        ]
        
        feedback = pseudo_relevance_feedback(results, top_k=2, weight=0.3)
        
        assert isinstance(feedback, dict)
        assert len(feedback) > 0
        # High-scoring terms should be present
        assert any("diabetes" in term or "glucose" in term for term in feedback.keys())
    
    def test_empty_results(self):
        """Test feedback with empty results."""
        feedback = pseudo_relevance_feedback([], top_k=3, weight=0.3)
        assert feedback == {}
    
    def test_string_payloads(self):
        """Test feedback with string payloads."""
        results = [
            ("diabetes treatment guidelines", 0.9),
            ("insulin therapy protocol", 0.8),
        ]
        
        feedback = pseudo_relevance_feedback(results, top_k=2, weight=0.3)
        
        assert isinstance(feedback, dict)


class TestAdaptiveFusion:
    """Tests for adaptive fusion."""
    
    def test_balanced_confidence(self):
        """Test fusion with balanced confidence."""
        dense_results = [("doc1", 0.8), ("doc2", 0.6), ("doc3", 0.4)]
        sparse_results = [("doc2", 0.7), ("doc3", 0.5), ("doc4", 0.3)]
        
        dense_w, sparse_w = adaptive_fusion_weights(
            dense_results,
            sparse_results,
            confidence_threshold=0.7
        )
        
        assert 0.0 <= dense_w <= 1.0
        assert 0.0 <= sparse_w <= 1.0
        assert abs(dense_w + sparse_w - 1.0) < 1e-6
    
    def test_high_dense_confidence(self):
        """Test fusion when dense has higher confidence."""
        dense_results = [("doc1", 0.95), ("doc2", 0.90), ("doc3", 0.85)]
        sparse_results = [("doc2", 0.4), ("doc3", 0.3), ("doc4", 0.2)]
        
        dense_w, sparse_w = adaptive_fusion_weights(
            dense_results,
            sparse_results,
            confidence_threshold=0.7
        )
        
        # Dense should get more weight
        assert dense_w > sparse_w
    
    def test_empty_results(self):
        """Test fusion with empty results."""
        dense_w, sparse_w = adaptive_fusion_weights([], [], confidence_threshold=0.7)
        
        assert dense_w == 0.5
        assert sparse_w == 0.5
    
    def test_only_dense_results(self):
        """Test fusion with only dense results."""
        dense_results = [("doc1", 0.9), ("doc2", 0.7)]
        
        dense_w, sparse_w = adaptive_fusion_weights(
            dense_results,
            [],
            confidence_threshold=0.7
        )
        
        assert dense_w == 1.0
        assert sparse_w == 0.0


class TestDiversityPenalty:
    """Tests for diversity penalty."""
    
    def test_apply_penalty(self):
        """Test applying diversity penalty."""
        results = [
            ({"text": "diabetes type 2 treatment guidelines"}, 0.9),
            ({"text": "diabetes type 2 management protocol"}, 0.85),
            ({"text": "heart disease prevention strategies"}, 0.8),
        ]
        
        reranked = apply_diversity_penalty(results, penalty=0.5, k=3)
        
        assert len(reranked) == 3
        # Third doc (different topic) should potentially rank higher
        doc_texts = [d["text"] for d, _ in reranked]
        assert "heart disease" in " ".join(doc_texts)
    
    def test_no_penalty(self):
        """Test with zero penalty."""
        results = [("doc1", 0.9), ("doc2", 0.8), ("doc3", 0.7)]
        
        reranked = apply_diversity_penalty(results, penalty=0.0, k=3)
        
        # Should be unchanged
        assert reranked == results
    
    def test_empty_results(self):
        """Test with empty results."""
        reranked = apply_diversity_penalty([], penalty=0.5, k=3)
        assert reranked == []


class TestThresholdFiltering:
    """Tests for threshold filtering."""
    
    def test_min_threshold(self):
        """Test minimum threshold filtering."""
        results = [
            ("doc1", 0.9),
            ("doc2", 0.6),
            ("doc3", 0.3),
        ]
        
        filtered = filter_by_similarity_thresholds(results, min_threshold=0.5)
        
        assert len(filtered) == 2
        assert filtered[0][0] == "doc1"
        assert filtered[1][0] == "doc2"
    
    def test_max_threshold(self):
        """Test maximum threshold filtering."""
        results = [
            ("doc1", 0.95),
            ("doc2", 0.7),
            ("doc3", 0.5),
        ]
        
        filtered = filter_by_similarity_thresholds(results, max_threshold=0.8)
        
        assert len(filtered) == 2
        assert filtered[0][0] == "doc2"
    
    def test_both_thresholds(self):
        """Test both min and max thresholds."""
        results = [
            ("doc1", 0.9),
            ("doc2", 0.7),
            ("doc3", 0.5),
            ("doc4", 0.3),
        ]
        
        filtered = filter_by_similarity_thresholds(
            results,
            min_threshold=0.4,
            max_threshold=0.8
        )
        
        assert len(filtered) == 2
        assert filtered[0][0] == "doc2"
        assert filtered[1][0] == "doc3"


class TestBordaCountFusion:
    """Tests for Borda count fusion."""
    
    def test_basic_fusion(self):
        """Test basic Borda count."""
        rank1 = [("doc1", 0.9), ("doc2", 0.7), ("doc3", 0.5)]
        rank2 = [("doc2", 0.8), ("doc1", 0.6), ("doc4", 0.4)]
        
        combined = borda_count_fusion([rank1, rank2], k=3)
        
        assert len(combined) == 3
        # Doc1 and doc2 appear in both, should rank high
        top_docs = [doc for doc, _ in combined[:2]]
        assert "doc1" in top_docs
        assert "doc2" in top_docs
    
    def test_single_ranking(self):
        """Test with single ranking."""
        rank1 = [("doc1", 0.9), ("doc2", 0.7)]
        
        combined = borda_count_fusion([rank1], k=2)
        
        assert len(combined) == 2
        assert combined[0][0] == "doc1"


class TestDistributionBasedFusion:
    """Tests for distribution-based fusion."""
    
    def test_z_score_fusion(self):
        """Test z-score normalization fusion."""
        dense = [("doc1", 0.9), ("doc2", 0.5), ("doc3", 0.3)]
        sparse = [("doc2", 10.0), ("doc3", 8.0), ("doc4", 2.0)]
        
        combined = distribution_based_fusion(dense, sparse, method="z_score", k=3)
        
        assert len(combined) <= 3
        # Doc2 appears in both, should rank high
        assert combined[0][0] == "doc2"
    
    def test_min_max_fusion(self):
        """Test min-max normalization fusion."""
        dense = [("doc1", 1.0), ("doc2", 0.5)]
        sparse = [("doc2", 100.0), ("doc3", 50.0)]
        
        combined = distribution_based_fusion(dense, sparse, method="min_max", k=3)
        
        assert len(combined) <= 3
    
    def test_rank_fusion(self):
        """Test rank-based fusion."""
        dense = [("doc1", 0.9), ("doc2", 0.7)]
        sparse = [("doc2", 10.0), ("doc3", 8.0)]
        
        combined = distribution_based_fusion(dense, sparse, method="rank", k=3)
        
        assert len(combined) <= 3
    
    def test_empty_results(self):
        """Test fusion with empty results."""
        combined = distribution_based_fusion([], [], method="z_score", k=5)
        assert combined == []


class TestVectorDBConfigValidation:
    """Tests for VectorDBConfig validation."""
    
    def test_valid_config(self):
        """Test creating valid config."""
        config = VectorDBConfig(
            multi_vector_enabled=True,
            vectors_per_doc=3,
            multi_vector_strategy="max_sim"
        )
        
        assert config.multi_vector_enabled is True
    
    def test_invalid_weight(self):
        """Test invalid weight validation."""
        with pytest.raises(ValueError, match="hybrid_dense_weight"):
            VectorDBConfig(hybrid_dense_weight=1.5)
    
    def test_invalid_multi_vector_strategy(self):
        """Test invalid multi-vector strategy."""
        with pytest.raises(ValueError, match="multi_vector_strategy"):
            VectorDBConfig(multi_vector_strategy="invalid")
    
    def test_invalid_expansion_method(self):
        """Test invalid query expansion method."""
        with pytest.raises(ValueError, match="query_expansion_method"):
            VectorDBConfig(query_expansion_method="invalid")
    
    def test_invalid_vectors_per_doc(self):
        """Test invalid vectors_per_doc."""
        with pytest.raises(ValueError, match="vectors_per_doc"):
            VectorDBConfig(multi_vector_enabled=True, vectors_per_doc=0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
