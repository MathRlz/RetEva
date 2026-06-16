"""Shared logging banner + formatting constants for stage handlers.

Extracted from the former ``evaluation/phased.py`` (Phase 1, X6) so the handler
submodules and the ``phased`` shim share one definition without importing each other
(avoids a circular import).
"""

from __future__ import annotations

from typing import Any, List

from ...logging_config import get_logger

logger = get_logger(__name__)


def publish_keyed_or_plain(
    s: Any, name: str, values: List, query_ids: List
) -> None:
    """Publish ``name`` as a keyed ``ItemSet`` when ``query_ids`` align 1:1 (W2/W3);
    otherwise the plain list (legacy path, e.g. a dataset with no usable ids). Either way
    ``get_artifact(name)`` returns the list, so positional consumers are unchanged; keyed
    consumers read it via ``get_items``. The single publish contract for every per-query
    transform output (query_text / optimized / refined / retrieved / …)."""
    ids = [str(i) for i in (query_ids or [])]
    if len(ids) == len(values) and len(set(ids)) == len(ids):
        from ..item_set import ItemSet

        s.put_items(name, ItemSet(ids, list(values)))
    else:
        s.put_artifact(name, values)

# Retrieval-debug formatting constants.
LOG_DIVIDER = "=" * 50
DEBUG_SAMPLE_LIMIT = 3
MATCH_SYMBOL = "✓"
MISS_SYMBOL = "✗"
