"""Hybrid retrieval fusion strategy registry."""

from typing import Any, Dict, List, Protocol, Tuple

from .rag.hybrid import reciprocal_rank_fusion
from ...constants import MIN_NORM_THRESHOLD


SearchResults = List[Tuple[Any, float]]


def _payload_key(payload: Any) -> str:
    if isinstance(payload, dict):
        if payload.get("doc_id") is not None:
            return str(payload["doc_id"])
        if payload.get("text") is not None:
            return str(payload["text"])
    return str(payload)


def _min_max_norm(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    vals = list(scores.values())
    mn, mx = min(vals), max(vals)
    if mx - mn < MIN_NORM_THRESHOLD:
        return {k: 1.0 for k in scores}
    return {k: (v - mn) / (mx - mn) for k, v in scores.items()}


class HybridFusionStrategy(Protocol):
    """Protocol for hybrid dense+sparse result fusion."""

    def fuse(
        self,
        dense_results: SearchResults,
        sparse_results: SearchResults,
        *,
        dense_weight: float,
        top_k: int,
        rrf_k: int,
    ) -> SearchResults:
        ...


class WeightedFusion:
    """Weighted linear combination over normalized scores."""

    def fuse(
        self,
        dense_results: SearchResults,
        sparse_results: SearchResults,
        *,
        dense_weight: float,
        top_k: int,
        rrf_k: int,
    ) -> SearchResults:
        dense_scores = {_payload_key(p): float(s) for p, s in dense_results}
        sparse_scores = {_payload_key(p): float(s) for p, s in sparse_results}
        dense_norm = _min_max_norm(dense_scores)
        sparse_norm = _min_max_norm(sparse_scores)

        payload_by_key = {}
        for payload, _ in dense_results:
            payload_by_key[_payload_key(payload)] = payload
        for payload, _ in sparse_results:
            payload_by_key[_payload_key(payload)] = payload

        merged = {}
        keys = set(dense_norm.keys()) | set(sparse_norm.keys())
        for key in keys:
            merged[key] = (
                dense_weight * dense_norm.get(key, 0.0)
                + (1.0 - dense_weight) * sparse_norm.get(key, 0.0)
            )

        ranked = sorted(merged.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [(payload_by_key[key], score) for key, score in ranked]


class RRFFusion:
    """Reciprocal Rank Fusion strategy."""

    def fuse(
        self,
        dense_results: SearchResults,
        sparse_results: SearchResults,
        *,
        dense_weight: float,
        top_k: int,
        rrf_k: int,
    ) -> SearchResults:
        return reciprocal_rank_fusion([dense_results, sparse_results], k=rrf_k, top_n=top_k)


class MaxScoreFusion:
    """Take max weighted normalized score per document."""

    def fuse(
        self,
        dense_results: SearchResults,
        sparse_results: SearchResults,
        *,
        dense_weight: float,
        top_k: int,
        rrf_k: int,
    ) -> SearchResults:
        dense_scores = {_payload_key(p): float(s) for p, s in dense_results}
        sparse_scores = {_payload_key(p): float(s) for p, s in sparse_results}
        dense_norm = _min_max_norm(dense_scores)
        sparse_norm = _min_max_norm(sparse_scores)

        payload_by_key = {}
        for payload, _ in dense_results:
            payload_by_key[_payload_key(payload)] = payload
        for payload, _ in sparse_results:
            payload_by_key[_payload_key(payload)] = payload

        merged = {}
        keys = set(dense_norm.keys()) | set(sparse_norm.keys())
        for key in keys:
            merged[key] = max(
                dense_weight * dense_norm.get(key, 0.0),
                (1.0 - dense_weight) * sparse_norm.get(key, 0.0),
            )

        ranked = sorted(merged.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [(payload_by_key[key], score) for key, score in ranked]


FUSION_REGISTRY: Dict[str, HybridFusionStrategy] = {
    "weighted": WeightedFusion(),
    "rrf": RRFFusion(),
    "max_score": MaxScoreFusion(),
}


def fuse_hybrid_results(
    method: str,
    dense_results: SearchResults,
    sparse_results: SearchResults,
    *,
    dense_weight: float,
    top_k: int,
    rrf_k: int,
) -> SearchResults:
    """Fuse hybrid results via registered strategy."""
    strategy = FUSION_REGISTRY.get(method)
    if strategy is None:
        raise ValueError(
            f"Unsupported hybrid fusion method: {method}. "
            f"Supported: {', '.join(sorted(FUSION_REGISTRY.keys()))}"
        )
    return strategy.fuse(
        dense_results,
        sparse_results,
        dense_weight=dense_weight,
        top_k=top_k,
        rrf_k=rrf_k,
    )

