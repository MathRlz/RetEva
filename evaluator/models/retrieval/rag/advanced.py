"""Advanced RAG techniques for improved retrieval.

This module implements advanced retrieval-augmented generation techniques:
- Multi-vector retrieval: Multiple embeddings per document
- Query expansion: Expanding queries with related terms
- Pseudo-relevance feedback: Using top results to refine query
- Adaptive fusion: Dynamically adjusting fusion strategies
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from collections import Counter

from ....logging_config import get_logger
from ....config import VectorDBConfig

logger = get_logger(__name__)


def multi_vector_search(
    query_embedding: np.ndarray,
    doc_vectors: List[List[np.ndarray]],
    doc_payloads: List[Any],
    strategy: str = "max_sim",
    k: int = 5
) -> List[Tuple[Any, float]]:
    """Search using multi-vector representations per document.
    
    Each document is represented by multiple vectors (e.g., sentence-level
    embeddings). Different strategies combine these for ranking.
    
    Args:
        query_embedding: Query vector.
        doc_vectors: List of vector lists (one list per document).
        doc_payloads: Document metadata/payloads.
        strategy: Combination strategy. Options: "max_sim", "avg_sim", "late_interaction".
        k: Number of results to return.
        
    Returns:
        List of (payload, score) tuples sorted by score.
        
    Examples:
        >>> query = np.array([0.1, 0.2, 0.3])
        >>> doc1_vectors = [np.array([0.1, 0.2, 0.3]), np.array([0.2, 0.3, 0.4])]
        >>> doc2_vectors = [np.array([0.9, 0.8, 0.7])]
        >>> results = multi_vector_search(query, [doc1_vectors, doc2_vectors], [doc1, doc2])
    """
    if len(doc_vectors) != len(doc_payloads):
        raise ValueError("doc_vectors and doc_payloads must have same length")
    
    scores = []
    
    for doc_vecs in doc_vectors:
        if not doc_vecs:
            scores.append(0.0)
            continue
        
        # Compute similarities between query and all document vectors
        doc_vecs_array = np.array(doc_vecs)
        
        # Normalize
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        docs_norm = doc_vecs_array / (np.linalg.norm(doc_vecs_array, axis=1, keepdims=True) + 1e-8)
        
        # Compute cosine similarities
        sims = np.dot(docs_norm, query_norm)
        
        if strategy == "max_sim":
            # MaxSim: Take maximum similarity across all vectors
            score = float(np.max(sims))
        elif strategy == "avg_sim":
            # Average similarity across all vectors
            score = float(np.mean(sims))
        elif strategy == "late_interaction":
            # Late interaction: Sum of top-k similarities
            top_k_sims = np.sort(sims)[-min(3, len(sims)):]
            score = float(np.sum(top_k_sims))
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        scores.append(score)
    
    # Sort by score and return top-k
    ranked = sorted(
        zip(doc_payloads, scores),
        key=lambda x: x[1],
        reverse=True
    )
    
    return ranked[:k]


def expand_query_with_synonyms(
    query: str,
    num_terms: int = 5
) -> str:
    """Expand query with synonyms (rule-based for medical domain).
    
    Args:
        query: Original query text.
        num_terms: Maximum number of expansion terms.
        
    Returns:
        Expanded query text.
    """
    # Simple medical synonym expansion
    # In production, this could use WordNet, medical ontologies, etc.
    medical_synonyms = {
        "htn": "hypertension",
        "dm": "diabetes mellitus",
        "mi": "myocardial infarction",
        "cvd": "cardiovascular disease",
        "copd": "chronic obstructive pulmonary disease",
        "ckd": "chronic kidney disease",
        "cad": "coronary artery disease",
        "chf": "congestive heart failure",
        "afib": "atrial fibrillation",
        "pe": "pulmonary embolism",
        "dvt": "deep vein thrombosis",
        "uti": "urinary tract infection",
        "gi": "gastrointestinal",
        "ct": "computed tomography",
        "mri": "magnetic resonance imaging",
    }
    
    expanded_terms = []
    query_lower = query.lower()
    
    for abbr, full_term in medical_synonyms.items():
        if abbr in query_lower.split():
            expanded_terms.append(full_term)
        if len(expanded_terms) >= num_terms:
            break
    
    if expanded_terms:
        return f"{query} {' '.join(expanded_terms)}"
    
    return query


def expand_query_with_embeddings(
    query_embedding: np.ndarray,
    corpus_embeddings: np.ndarray,
    corpus_terms: List[str],
    num_terms: int = 5
) -> List[str]:
    """Expand query using nearest terms in embedding space.
    
    Args:
        query_embedding: Query vector.
        corpus_embeddings: All term embeddings.
        corpus_terms: Corresponding terms.
        num_terms: Number of expansion terms.
        
    Returns:
        List of expansion terms.
    """
    # Normalize embeddings
    query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
    corpus_norm = corpus_embeddings / (np.linalg.norm(corpus_embeddings, axis=1, keepdims=True) + 1e-8)
    
    # Compute similarities
    sims = np.dot(corpus_norm, query_norm)
    
    # Get top-k most similar terms
    top_indices = np.argsort(sims)[-num_terms:][::-1]
    
    return [corpus_terms[i] for i in top_indices]


def pseudo_relevance_feedback(
    initial_results: List[Tuple[Any, float]],
    top_k: int = 3,
    weight: float = 0.3
) -> Dict[str, float]:
    """Extract feedback terms from top retrieved documents.
    
    Implements pseudo-relevance feedback by extracting important terms
    from top-k results to refine the query.
    
    Args:
        initial_results: Initial retrieval results (payload, score).
        top_k: Number of top documents to use.
        weight: Weight for feedback terms.
        
    Returns:
        Dictionary of feedback terms and their weights.
    """
    if not initial_results:
        return {}
    
    # Extract text from top-k results
    top_docs = initial_results[:top_k]
    feedback_terms = Counter()
    
    for payload, score in top_docs:
        # Extract text from payload
        if isinstance(payload, dict):
            text = payload.get("text", "") or payload.get("content", "")
        else:
            text = str(payload)
        
        # Tokenize and count (simple whitespace tokenization)
        tokens = [
            t.strip().lower()
            for t in text.split()
            if len(t.strip()) > 3  # Filter short words
        ]
        
        # Weight by document score
        for token in tokens:
            feedback_terms[token] += score
    
    # Normalize and apply weight
    if feedback_terms:
        max_count = max(feedback_terms.values())
        normalized_terms = {
            term: (count / max_count) * weight
            for term, count in feedback_terms.items()
        }
        
        # Sort by weight and return top 10
        sorted_terms = sorted(
            normalized_terms.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return dict(sorted_terms[:10])
    
    return {}


def adaptive_fusion_weights(
    dense_results: List[Tuple[Any, float]],
    sparse_results: List[Tuple[Any, float]],
    confidence_threshold: float = 0.7
) -> Tuple[float, float]:
    """Adaptively determine fusion weights based on result confidence.
    
    Analyzes the confidence/quality of dense and sparse results to
    dynamically adjust fusion weights.
    
    Args:
        dense_results: Dense retrieval results.
        sparse_results: Sparse retrieval results.
        confidence_threshold: Threshold for high confidence.
        
    Returns:
        Tuple of (dense_weight, sparse_weight) that sum to 1.0.
    """
    if not dense_results and not sparse_results:
        return 0.5, 0.5
    
    if not dense_results:
        return 0.0, 1.0
    
    if not sparse_results:
        return 1.0, 0.0
    
    # Compute confidence metrics
    # Use score distribution and top score magnitude
    dense_scores = [score for _, score in dense_results[:5]]
    sparse_scores = [score for _, score in sparse_results[:5]]
    
    # Confidence = top score magnitude * score concentration
    def compute_confidence(scores):
        if not scores:
            return 0.0
        
        top_score = max(scores)
        score_std = np.std(scores) if len(scores) > 1 else 0.0
        
        # Higher top score = more confident
        # Lower std = more concentrated at top = more confident
        concentration = 1.0 / (1.0 + score_std)
        
        return top_score * concentration
    
    dense_conf = compute_confidence(dense_scores)
    sparse_conf = compute_confidence(sparse_scores)
    
    # Normalize to weights
    total_conf = dense_conf + sparse_conf
    
    if total_conf > 0:
        dense_weight = dense_conf / total_conf
        sparse_weight = sparse_conf / total_conf
    else:
        dense_weight = 0.5
        sparse_weight = 0.5
    
    logger.debug(
        f"Adaptive fusion: dense_conf={dense_conf:.3f}, sparse_conf={sparse_conf:.3f}, "
        f"weights=({dense_weight:.3f}, {sparse_weight:.3f})"
    )
    
    return dense_weight, sparse_weight


def apply_diversity_penalty(
    results: List[Tuple[Any, float]],
    penalty: float = 0.1,
    k: int = None
) -> List[Tuple[Any, float]]:
    """Apply diversity penalty to reduce redundancy in results.
    
    Penalizes documents that are too similar to higher-ranked documents.
    
    Args:
        results: Retrieved results (payload, score).
        penalty: Penalty factor (0.0-1.0).
        k: Number of results to return (None = all).
        
    Returns:
        Re-ranked results with diversity penalty applied.
    """
    if penalty <= 0.0 or not results:
        return results[:k] if k else results
    
    # Extract texts for similarity computation
    def get_text(payload):
        if isinstance(payload, dict):
            return payload.get("text", "") or payload.get("content", "")
        return str(payload)
    
    texts = [get_text(payload) for payload, _ in results]
    
    # Simple token-based similarity
    def token_similarity(text1, text2):
        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())
        
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        
        return intersection / union if union > 0 else 0.0
    
    # Re-rank with diversity penalty
    reranked = []
    selected_texts = []
    
    for payload, score in results:
        text = get_text(payload)
        
        # Compute max similarity to already selected documents
        max_sim = 0.0
        for selected_text in selected_texts:
            sim = token_similarity(text, selected_text)
            max_sim = max(max_sim, sim)
        
        # Apply penalty
        penalized_score = score * (1.0 - penalty * max_sim)
        
        reranked.append((payload, penalized_score))
        selected_texts.append(text)
    
    # Re-sort by penalized scores
    reranked.sort(key=lambda x: x[1], reverse=True)
    
    return reranked[:k] if k else reranked


def filter_by_similarity_thresholds(
    results: List[Tuple[Any, float]],
    min_threshold: Optional[float] = None,
    max_threshold: Optional[float] = None
) -> List[Tuple[Any, float]]:
    """Filter results by similarity score thresholds.
    
    Args:
        results: Retrieved results (payload, score).
        min_threshold: Minimum score threshold (inclusive).
        max_threshold: Maximum score threshold (exclusive).
        
    Returns:
        Filtered results.
    """
    filtered = results
    
    if min_threshold is not None:
        filtered = [(p, s) for p, s in filtered if s >= min_threshold]
    
    if max_threshold is not None:
        filtered = [(p, s) for p, s in filtered if s < max_threshold]
    
    return filtered
