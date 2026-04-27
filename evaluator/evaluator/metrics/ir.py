from typing import List, Dict
import numpy as np
import logging

logger = logging.getLogger(__name__)

def reciprocal_rank(retrieved: List[str], relevant: Dict[str, int]) -> float:
    """Compute reciprocal rank of first relevant document.
    
    Args:
        retrieved: List of retrieved document IDs in rank order
        relevant: Dict mapping doc IDs to relevance scores (>0 = relevant)
        
    Returns:
        Reciprocal rank (1/position of first relevant doc, or 0 if none)
    """
    if not retrieved:
        return 0.0
    if not relevant:
        logger.warning("Empty relevant dict, RR will be 0")
        return 0.0
    
    for i, doc_id in enumerate(retrieved, 1):
        if relevant.get(doc_id, 0) > 0:
            return 1.0 / i
    return 0.0

def precision_at_k(retrieved: List[str], relevant: Dict[str, int], k: int) -> float:
    """Compute precision at rank k.
    
    Args:
        retrieved: List of retrieved document IDs
        relevant: Dict mapping doc IDs to relevance scores
        k: Cutoff rank
        
    Returns:
        Fraction of top-k results that are relevant
    """
    if k <= 0:
        logger.warning(f"Invalid k={k}, returning 0")
        return 0.0
    return sum(1 for doc_id in retrieved[:k] if relevant.get(doc_id, 0) > 0) / k

def recall_at_k(retrieved: List[str], relevant: Dict[str, int], k: int) -> float:
    """Compute recall at rank k.
    
    Args:
        retrieved: List of retrieved document IDs
        relevant: Dict mapping doc IDs to relevance scores
        k: Cutoff rank
        
    Returns:
        Fraction of relevant docs found in top-k results
    """
    if k <= 0:
        logger.warning(f"Invalid k={k}, returning 0")
        return 0.0
    
    total = sum(1 for v in relevant.values() if v > 0)
    if total == 0:
        logger.warning("No relevant documents, recall is 0")
        return 0.0
    return sum(1 for doc_id in retrieved[:k] if relevant.get(doc_id, 0) > 0) / total

def dcg_at_k(retrieved: List[str], relevant: Dict[str, int], k: int) -> float:
    """Compute Discounted Cumulative Gain at rank k.
    
    Args:
        retrieved: List of retrieved document IDs
        relevant: Dict mapping doc IDs to relevance scores
        k: Cutoff rank
        
    Returns:
        DCG@k score
    """
    if k <= 0 or not retrieved:
        return 0.0
    
    return sum((2 ** relevant.get(doc_id, 0) - 1) / np.log2(i + 2)
               for i, doc_id in enumerate(retrieved[:k]))

def ndcg_at_k(retrieved: List[str], relevant: Dict[str, int], k: int) -> float:
    """Compute Normalized Discounted Cumulative Gain at rank k.
    
    Args:
        retrieved: List of retrieved document IDs
        relevant: Dict mapping doc IDs to relevance scores
        k: Cutoff rank
        
    Returns:
        NDCG@k score (0-1, higher is better)
    """
    if k <= 0:
        logger.warning(f"Invalid k={k}, returning 0")
        return 0.0
    
    dcg = dcg_at_k(retrieved, relevant, k)
    ideal = sorted([v for v in relevant.values() if v > 0], reverse=True)
    
    if not ideal:
        logger.warning("No relevant documents for NDCG computation")
        return 0.0
    
    idcg = sum((2 ** rel - 1) / np.log2(i + 2) for i, rel in enumerate(ideal[:k]))
    return dcg / idcg if idcg > 0 else 0.0

def average_precision(retrieved: List[str], relevant: Dict[str, int]) -> float:
    """Compute Average Precision.
    
    Args:
        retrieved: List of retrieved document IDs
        relevant: Dict mapping doc IDs to relevance scores
        
    Returns:
        Average precision score
    """
    if not retrieved:
        return 0.0
    
    total = sum(1 for v in relevant.values() if v > 0)
    if total == 0:
        logger.warning("No relevant documents for AP computation")
        return 0.0
    
    score, hit = 0.0, 0
    for i, doc_id in enumerate(retrieved, 1):
        if relevant.get(doc_id, 0) > 0:
            hit += 1
            score += hit / i
    return score / total