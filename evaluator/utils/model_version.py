"""Resolve a model's *version* for cache-key invalidation (task T3).

Caching keys on ``model_name`` alone is unsound: swap the weights behind the same name (a new
HF revision, a fine-tune checkpoint) and a stale artifact is silently reused → wrong numbers. A
version folded into the key fixes that. Models aren't required to expose a version, so this is a
best-effort probe with a stable order of preference; ``None`` means "no version known" and leaves
the legacy key unchanged (back-compat — existing caches still hit).
"""

from __future__ import annotations

from typing import Any, Optional


def resolve_model_version(model: Any) -> Optional[str]:
    """Best-effort version/revision string for ``model`` (``None`` if unknown).

    Order of preference:
    1. an explicit ``model.version()`` / ``model.model_version`` (a family that knows its rev);
    2. the HuggingFace commit hash on the wrapped ``transformers`` config / module;
    3. an explicit ``revision`` attribute.
    """
    if model is None:
        return None

    versioner = getattr(model, "version", None)
    if callable(versioner):
        try:
            v = versioner()
            if v:
                return str(v)
        except Exception:
            pass

    for attr in ("model_version", "revision", "_commit_hash"):
        v = getattr(model, attr, None)
        if v:
            return str(v)

    # Wrapped HF module/config commit hash (set when loaded from the hub at a pinned rev).
    inner = getattr(model, "model", None)
    for holder in (
        inner,
        getattr(inner, "config", None),
        getattr(model, "config", None),
    ):
        v = getattr(holder, "_commit_hash", None)
        if v:
            return str(v)

    return None
