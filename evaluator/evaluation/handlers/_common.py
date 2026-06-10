"""Shared logging banner + formatting constants for stage handlers.

Extracted from the former ``evaluation/phased.py`` (Phase 1, X6) so the handler
submodules and the ``phased`` shim share one definition without importing each other
(avoids a circular import).
"""

from __future__ import annotations

from ...logging_config import get_logger

logger = get_logger(__name__)

# Retrieval-debug formatting constants.
LOG_DIVIDER = "=" * 50
DEBUG_SAMPLE_LIMIT = 3
MATCH_SYMBOL = "✓"
MISS_SYMBOL = "✗"
