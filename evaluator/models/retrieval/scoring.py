"""Shared payload/score primitives for retrieval strategies.

Single home for the helpers that were hand-copied between the retrieval
pipeline and the fusion registry: payload text/key extraction, whitespace
tokenization, and min-max score normalization.
"""

from __future__ import annotations

from typing import Any, Dict, List, TypeVar

from ...constants import MIN_NORM_THRESHOLD

K = TypeVar("K")


def payload_text(payload: Any) -> str:
    """The searchable text of a payload (dict ``text`` field or str())."""
    if isinstance(payload, dict):
        return str(payload.get("text", ""))
    return str(payload)


def payload_key(payload: Any) -> str:
    """Stable identity of a payload for cross-list score merging (doc_id > text > str)."""
    if isinstance(payload, dict):
        if payload.get("doc_id") is not None:
            return str(payload["doc_id"])
        if payload.get("text") is not None:
            return str(payload["text"])
    return str(payload)


def tokenize(text: str) -> List[str]:
    """Lowercase whitespace tokenization (BM25 / token-overlap shared)."""
    return [tok for tok in text.lower().split() if tok]


def min_max_norm(scores: Dict[K, float]) -> Dict[K, float]:
    """Min-max normalize to [0, 1]; constant lists map to all-1.0."""
    if not scores:
        return {}
    vals = list(scores.values())
    mn, mx = min(vals), max(vals)
    if mx - mn < MIN_NORM_THRESHOLD:
        return {k: 1.0 for k in scores}
    return {k: (v - mn) / (mx - mn) for k, v in scores.items()}
