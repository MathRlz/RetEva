"""Hybrid retrieval fusion strategy registry."""

from typing import Any, Dict, List, Protocol, Tuple

from .rag.hybrid import reciprocal_rank_fusion
from .scoring import min_max_norm as _min_max_norm, payload_key as _payload_key


SearchResults = List[Tuple[Any, float]]


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


def _normalize_hybrid(dense_results, sparse_results):
    """Shared setup for the weighted/max strategies: each list's min-max-normalized score dict
    (keyed by payload) + the payload-by-key lookup (last write wins). The two strategies differ
    only in how they merge ``dense_norm``/``sparse_norm`` afterwards."""
    dense_norm = _min_max_norm({_payload_key(p): float(s) for p, s in dense_results})
    sparse_norm = _min_max_norm({_payload_key(p): float(s) for p, s in sparse_results})
    payload_by_key = {}
    for payload, _ in dense_results:
        payload_by_key[_payload_key(payload)] = payload
    for payload, _ in sparse_results:
        payload_by_key[_payload_key(payload)] = payload
    return dense_norm, sparse_norm, payload_by_key


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
        dense_norm, sparse_norm, payload_by_key = _normalize_hybrid(
            dense_results, sparse_results
        )
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
        dense_norm, sparse_norm, payload_by_key = _normalize_hybrid(
            dense_results, sparse_results
        )
        merged = {}
        keys = set(dense_norm.keys()) | set(sparse_norm.keys())
        for key in keys:
            merged[key] = max(
                dense_weight * dense_norm.get(key, 0.0),
                (1.0 - dense_weight) * sparse_norm.get(key, 0.0),
            )

        ranked = sorted(merged.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [(payload_by_key[key], score) for key, score in ranked]


# The registry store (kept public as ``FUSION_REGISTRY`` for back-compat); populate via
# ``register_fusion`` so a plugin can add a strategy without editing this module — the OCP
# extension point, matching the model / node / metric / vector-store registries.
FUSION_REGISTRY: Dict[str, HybridFusionStrategy] = {}


def register_fusion(name: str, strategy: HybridFusionStrategy) -> None:
    """Register a hybrid dense+sparse fusion strategy under ``name``."""
    FUSION_REGISTRY[name] = strategy


def list_fusions() -> List[str]:
    """The registered hybrid-fusion method names (sorted) — the single source for config
    validation and the builder UI's `method` select."""
    return sorted(FUSION_REGISTRY)


register_fusion("weighted", WeightedFusion())
register_fusion("rrf", RRFFusion())
register_fusion("max_score", MaxScoreFusion())


def fuse_hybrid_results(
    method: str,
    dense_results: SearchResults,
    sparse_results: SearchResults,
    *,
    dense_weight: float,
    top_k: int,
    rrf_k: int,
) -> SearchResults:
    """Fuse hybrid dense+sparse results via the registered strategy ``method``."""
    strategy = FUSION_REGISTRY.get(method)
    if strategy is None:
        raise ValueError(
            f"Unsupported hybrid fusion method: {method}. Supported: {', '.join(list_fusions())}"
        )
    return strategy.fuse(
        dense_results,
        sparse_results,
        dense_weight=dense_weight,
        top_k=top_k,
        rrf_k=rrf_k,
    )
