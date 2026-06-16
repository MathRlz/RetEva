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

from typing import Any, Optional


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
