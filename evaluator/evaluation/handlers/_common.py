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
    consumers read it via ``keyed_items``. The single publish contract for every per-query
    transform output (query_text / optimized / refined / retrieved / …)."""
    ids = [str(i) for i in (query_ids or [])]
    if len(ids) == len(values) and len(set(ids)) == len(ids):
        from ..item_set import ItemSet

        s.put_items(name, ItemSet(ids, list(values)))
    else:
        s.put_artifact(name, values)


# ── node-local graph signals (graph-truth Phase 4) ────────────────────────────────────
# Handlers ask about THIS node's bindings + the executed graph's node kinds, never about
# run-global pipeline presence (the former retrieval_ran/asr_ran/is_asr_text_retrieval):
# those answered a per-branch question with a whole-run answer and misfired in every mixed
# graph (a text arm next to an ASR arm took the ASR code path and crashed). The mode string
# ``s.mode`` survives ONLY as the report's label — behavioral reads of it are banned.
def bound_producer_kind(s: Any, name: str) -> Any:
    """The node KIND of the producer bound (and published) for input ``name`` on the
    current node, or None. Pure graph truth: bindings say who feeds this node, the
    executed graph's ``node_kinds`` map says what that producer is."""
    kinds = getattr(s, "node_kinds", None) or {}
    for pid in reversed(s._producers(name)):
        if s.ctx.has(pid, name):
            return kinds.get(pid)
    return None


def asr_hypothesis(s: Any) -> list:
    """This node's bound ``query_text`` — but only when an ASR node produced it (the
    hypothesis WER/CER score against). Empty for a node fed by dataset/reference text:
    scoring reference-vs-reference would report a meaningless WER of 0."""
    if bound_producer_kind(s, "query_text") != "asr":
        return []
    return list(s.get_artifact("query_text", default=[]))


def _ctx_first(s: Any, name: str) -> Any:
    """The first bound-and-published value for input ``name`` — oldest producer first
    (shared-GT reads). binding-scoped (the consumer must DECLARE the read), no
    global bus scan — so graph isolation holds and streaming's lifetime analysis sees
    every terminal read."""
    for pid in s._producers(name):
        if s.ctx.has(pid, name):
            return s.ctx.get(pid, name)
    return None


def _reference_transcriptions(s: Any) -> list:
    """The spoken-transcription GT from the bus (`reference_transcription`, published by
    the asr / audio_embedding node, M1c-3; bus-only since M1d-2)."""
    return list(s.get_artifact("reference_transcription", default=[]))


def _relevant_from_bus(s: Any) -> list:
    """Per-query relevance from the bus (`relevant_docs`, dataset order), with the
    self-retrieval fallback (each spoken reference is its own relevant key). The single
    source the retrieval-metrics / answer / trace / judge consumers share."""
    return list(s.get_artifact("relevant_docs", default=[])) or [
        {str(gt): 1} for gt in _reference_transcriptions(s)
    ]


# Retrieval-debug formatting constants.
DEBUG_SAMPLE_LIMIT = 3
MATCH_SYMBOL = "✓"
MISS_SYMBOL = "✗"
