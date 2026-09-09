from typing import Any, List, Dict
import numpy as np
import logging

from ..constants import RETRIEVAL_DEPTH

logger = logging.getLogger(__name__)


def report_depth(k: Any) -> int:
    """Floor a configured retrieval depth at :data:`RETRIEVAL_DEPTH`."""
    return max(int(k), RETRIEVAL_DEPTH)


def reciprocal_rank(retrieved: List[str], relevant: Dict[str, int]) -> float:
    """Reciprocal rank of the first relevant doc (1/position); 0.0 if none (score > 0 counts
    as relevant)."""
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
    """Precision at k: fraction of the top-k retrieved docs that are relevant."""
    if k <= 0:
        logger.warning(f"Invalid k={k}, returning 0")
        return 0.0
    return sum(1 for doc_id in retrieved[:k] if relevant.get(doc_id, 0) > 0) / k


def recall_at_k(retrieved: List[str], relevant: Dict[str, int], k: int) -> float:
    """Recall at k: fraction of relevant docs found in the top-k results."""
    if k <= 0:
        logger.warning(f"Invalid k={k}, returning 0")
        return 0.0

    total = sum(1 for v in relevant.values() if v > 0)
    if total == 0:
        logger.warning("No relevant documents, recall is 0")
        return 0.0
    return sum(1 for doc_id in retrieved[:k] if relevant.get(doc_id, 0) > 0) / total


def dcg_at_k(retrieved: List[str], relevant: Dict[str, int], k: int) -> float:
    """Discounted Cumulative Gain at rank k: linear gain ``rel / log2(rank+1)`` (Järvelin;
    trec_eval's ``ndcg_cut`` definition, for comparability with BEIR/pytrec_eval numbers)."""
    # cross-checked per query against pytrec_eval in tests/test_ir_reference_impl.py
    if k <= 0 or not retrieved:
        return 0.0

    return sum(relevant.get(doc_id, 0) / np.log2(i + 2)
               for i, doc_id in enumerate(retrieved[:k]))


def ndcg_at_k(retrieved: List[str], relevant: Dict[str, int], k: int) -> float:
    """Normalized Discounted Cumulative Gain at rank k, in [0, 1] (higher is better)."""
    if k <= 0:
        logger.warning(f"Invalid k={k}, returning 0")
        return 0.0

    dcg = dcg_at_k(retrieved, relevant, k)
    ideal = sorted([v for v in relevant.values() if v > 0], reverse=True)

    if not ideal:
        logger.warning("No relevant documents for NDCG computation")
        return 0.0

    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal[:k]))
    return dcg / idcg if idcg > 0 else 0.0


def average_precision(retrieved: List[str], relevant: Dict[str, int]) -> float:
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
