"""Embedding-space identity (architecture A2).

A vector is only comparable to another from the **same embedding space** (same
model/projection). Each embedder declares an ``embedding_space`` via registry
metadata; absent that, the space defaults to a unique id per ``(model_type,
model_name)`` so distinct models never accidentally compare equal.

The *validators* that enforce this at pre-flight live in
``evaluation/validation.py`` (they are validation, not model construction);
this module owns only the registry-side identity resolution.

See ``evaluator-architecture.md`` §4.
"""

from __future__ import annotations

from typing import Any, FrozenSet, Optional, Set


def resolve_embedding_space(
    registry: Any, model_type: str, model_name: Optional[str] = None
) -> str:
    """The embedding-space id for a model: declared ``embedding_space`` metadata if present,
    else a unique fallback ``"<type>:<name>"`` (distinct models never collide)."""
    declared = registry.get_metadata(model_type).get("embedding_space")
    if declared:
        return str(declared)
    name = model_name or registry.get_default_name(model_type) or ""
    return f"{model_type}:{name}"


# ── Cross-modal compatible-space registry (Roadmap 2b) ───────────────
# Two DISTINCT space ids declared cross-comparable: their vectors live in the same geometry
# even though they are registered under different ids (e.g. the audio and text encoders a
# CLAP-style model projects into one shared contrastive space). Same-id spaces are always
# compatible and need no entry. Model authors / plugins populate this via
# ``register_compatible_spaces`` — the declarative replacement for hard-coding a model pair.
_COMPATIBLE_SPACES: Set[FrozenSet[str]] = set()


class EmbeddingSpaceMismatch(RuntimeError):
    """An index and a query were compared across incompatible embedding spaces."""


def register_compatible_spaces(a: str, b: str) -> None:
    """Declare two distinct embedding-space ids cross-comparable (idempotent). No-op for
    equal/empty ids — identical spaces are compatible by definition."""
    if a and b and a != b:
        _COMPATIBLE_SPACES.add(frozenset((a, b)))


def spaces_compatible(a: Optional[str], b: Optional[str]) -> bool:
    """Whether vectors from spaces ``a`` and ``b`` may be dotted. True when either is unknown
    (``None`` → unchecked), they are identical, or they are registered compatible."""
    if a is None or b is None or a == b:
        return True
    return frozenset((a, b)) in _COMPATIBLE_SPACES


def assert_spaces_compatible(
    index_space: Optional[str], query_space: Optional[str], *, where: str
) -> None:
    """Raise :class:`EmbeddingSpaceMismatch` if two *known* spaces are incompatible.

    No-op when either side is unknown (``None``) or the spaces are compatible — so it never
    rejects a valid same-space run (a loud error in place of a silent wrong answer)."""
    if not spaces_compatible(index_space, query_space):
        raise EmbeddingSpaceMismatch(
            f"Embedding-space mismatch at {where}: query vectors are in space "
            f"'{query_space}' but the index holds space '{index_space}'. Dense retrieval dots "
            f"these vectors, so the scores would be meaningless. Align the embedders, declare "
            f"the spaces compatible via register_compatible_spaces(), or use sparse retrieval."
        )
