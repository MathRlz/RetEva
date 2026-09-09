"""Resolve a model's *version* for cache-key invalidation.

Keying on ``model_name`` alone is unsound: swapped weights behind the same name (new HF
revision, fine-tune) would silently reuse a stale cached artifact. ``None`` means no version
was found and the unversioned key is used (existing caches still hit).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def resolve_model_version(model: Any) -> Optional[str]:
    """Best-effort version/revision string for ``model`` (``None`` if unknown); no single
    attribute is guaranteed, so several likely spots are probed in order, including the
    wrapped HF module's commit hash."""
    if model is None:
        return None

    versioner = getattr(model, "version", None)
    if callable(versioner):
        try:
            v = versioner()
            if v:
                return str(v)
        except Exception as exc:  # noqa: BLE001 - version probing is best-effort
            logger.debug("model version probe failed: %s", exc)

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
