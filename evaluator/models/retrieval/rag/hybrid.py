"""Reciprocal-rank fusion of multiple rankings."""

from typing import Any, Dict, List, Optional, Tuple

from ..scoring import payload_key


def reciprocal_rank_fusion(
    rankings: List[List[Tuple[Any, float]]],
    k: int = 60,
    top_n: Optional[int] = None
) -> List[Tuple[Any, float]]:
    """Combine multiple rankings using Reciprocal Rank Fusion (RRF).

    RRF score for document d = sum over rankings of 1 / (k + rank(d))

    This method is effective for combining rankings from different retrieval
    systems without requiring score normalization.

    Args:
        rankings: List of ranked result lists. Each list contains (item, score) tuples.
        k: RRF parameter controlling the impact of lower-ranked documents.
            Higher k gives more weight to top results. Default is 60.
        top_n: Number of top results to return. If None, returns all.

    Returns:
        List of (item, rrf_score) tuples sorted by RRF score descending.
    """
    rrf_scores: Dict[Any, float] = {}
    item_lookup: Dict[Any, Any] = {}

    for ranking in rankings:
        for rank, (item, _score) in enumerate(ranking, start=1):
            item_key = payload_key(item)
            rrf_scores[item_key] = rrf_scores.get(item_key, 0.0) + 1.0 / (k + rank)
            item_lookup[item_key] = item

    # Sort by RRF score descending
    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    if top_n is not None:
        sorted_items = sorted_items[:top_n]

    return [(item_lookup[key], score) for key, score in sorted_items]
