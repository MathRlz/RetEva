"""Shared logging banner + formatting constants for stage handlers.

A separate module so the handler submodules share one definition without importing each
other (avoids a circular import).
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


# ── graph-derived run signals (graph-first Phase 3) ───────────────────────────────────
# Behavioral branches read these instead of the ``pipeline_mode`` string, so the graph (what
# pipelines/nodes actually run) drives behavior — a mode-less graph run behaves correctly too.
# The report's mode *label* still rides ``s.mode`` (the executed graph's identity), since
# audio_emb vs audio_text is a run policy the node set can't express (identical graphs).
def retrieval_ran(s: Any) -> bool:
    """A retrieval (search) pipeline is part of this run. Replaces the former
    ``s.mode != 'asr_only'`` — ``asr_only`` is exactly the no-retrieval mode."""
    return s.retrieval_pipeline is not None


def asr_ran(s: Any) -> bool:
    """An ASR pipeline is part of this run (asr_text_retrieval / asr_only). Replaces
    ``s.mode in ('asr_text_retrieval', 'asr_only')`` — the audio modes carry no ASR."""
    return s.asr_pipeline is not None


def is_asr_text_retrieval(s: Any) -> bool:
    """ASR feeds retrieval → the effective query text is the ASR hypothesis (vs the spoken
    reference in audio modes). Replaces ``s.mode == 'asr_text_retrieval'`` = ASR ran AND
    retrieval ran (``asr_only`` has ASR but no retrieval)."""
    return s.asr_pipeline is not None and s.retrieval_pipeline is not None


def _ctx_first(s: Any, name: str) -> Any:
    """The first ``ItemSet`` published anywhere for artifact ``name`` (shared GT)."""
    for pid, art in s.ctx.slots():
        if art == name:
            return s.ctx.get(pid, name)
    return None


def _reference_transcriptions(s: Any) -> list:
    """The spoken-transcription GT from the bus (`reference_transcription`, published by
    the asr / audio_embedding node, M1c-3; bus-only since M1d-2)."""
    return list(s.get_artifact("reference_transcription", default=[]))


# Retrieval-debug formatting constants.
LOG_DIVIDER = "=" * 50
DEBUG_SAMPLE_LIMIT = 3
MATCH_SYMBOL = "✓"
MISS_SYMBOL = "✗"
