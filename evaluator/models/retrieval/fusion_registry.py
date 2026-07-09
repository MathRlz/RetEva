"""Hybrid retrieval fusion strategy registry."""

from typing import Any, Callable, Dict, List, Tuple

from .rag.hybrid import reciprocal_rank_fusion
from .scoring import min_max_norm as _min_max_norm, payload_key as _payload_key


SearchResults = List[Tuple[Any, float]]
FusionFn = Callable[..., SearchResults]


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


def _weighted_fusion(dense_results, sparse_results, *, dense_weight, top_k, rrf_k):
    """Weighted linear combination over normalized scores."""
    dense_norm, sparse_norm, payload_by_key = _normalize_hybrid(dense_results, sparse_results)
    merged = {}
    for key in set(dense_norm.keys()) | set(sparse_norm.keys()):
        merged[key] = (dense_weight * dense_norm.get(key, 0.0)
                       + (1.0 - dense_weight) * sparse_norm.get(key, 0.0))
    ranked = sorted(merged.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [(payload_by_key[key], score) for key, score in ranked]


def _rrf_fusion(dense_results, sparse_results, *, dense_weight, top_k, rrf_k):
    """Reciprocal Rank Fusion."""
    return reciprocal_rank_fusion([dense_results, sparse_results], k=rrf_k, top_n=top_k)


def _max_score_fusion(dense_results, sparse_results, *, dense_weight, top_k, rrf_k):
    """Take max weighted normalized score per document."""
    dense_norm, sparse_norm, payload_by_key = _normalize_hybrid(dense_results, sparse_results)
    merged = {}
    for key in set(dense_norm.keys()) | set(sparse_norm.keys()):
        merged[key] = max(dense_weight * dense_norm.get(key, 0.0),
                          (1.0 - dense_weight) * sparse_norm.get(key, 0.0))
    ranked = sorted(merged.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [(payload_by_key[key], score) for key, score in ranked]


# OCP extension point (matches the model / node / metric / vector-store registries): a plugin can
# register a fusion fn under a name without editing this module. Public as ``FUSION_REGISTRY``.
FUSION_REGISTRY: Dict[str, FusionFn] = {
    "weighted": _weighted_fusion,
    "rrf": _rrf_fusion,
    "max_score": _max_score_fusion,
}


def register_fusion(name: str, fn: FusionFn) -> None:
    """Register a hybrid dense+sparse fusion function under ``name``."""
    FUSION_REGISTRY[name] = fn


def list_fusions() -> List[str]:
    """The registered hybrid-fusion method names (sorted) — the single source for config
    validation and the builder UI's `method` select."""
    return sorted(FUSION_REGISTRY)


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
    fn = FUSION_REGISTRY.get(method)
    if fn is None:
        raise ValueError(
            f"Unsupported hybrid fusion method: {method}. Supported: {', '.join(list_fusions())}"
        )
    return fn(
        dense_results, sparse_results,
        dense_weight=dense_weight, top_k=top_k, rrf_k=rrf_k,
    )
