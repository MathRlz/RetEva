"""Shared logging banner + formatting constants for stage handlers.

A separate module so the handler submodules share one definition without importing each
other (avoids a circular import).
"""

from __future__ import annotations

from typing import Any, List

from ...logging_config import get_logger
from ..item_set import ItemSet

logger = get_logger(__name__)


def publish_keyed_or_plain(
    s: Any, name: str, values: List, query_ids: List
) -> None:
    """Publish ``name`` as a keyed ``ItemSet`` when ``query_ids`` align 1:1, else the plain
    list (e.g. a dataset with no usable ids). ``get_artifact(name)`` returns the list either
    way, so positional consumers are unchanged; keyed consumers read it via ``keyed_items``."""
    ids = [str(i) for i in (query_ids or [])]
    if len(ids) == len(values) and len(set(ids)) == len(ids):
        s.put_items(name, ItemSet(ids, list(values)))
    else:
        s.put_artifact(name, values)


# Handlers ask about THIS node's bindings + the executed graph's node kinds, never run-global
# pipeline presence — a per-branch question answered with a whole-run answer misfires in any
# mixed graph. ``s.mode`` survives only as the report's label; behavioral reads of it are banned.
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
    """First bound-and-published value for ``name`` (oldest producer first, shared-GT reads);
    binding-scoped with no global bus scan, so graph isolation holds and streaming's lifetime
    analysis sees every terminal read."""
    for pid in s._producers(name):
        if s.ctx.has(pid, name):
            return s.ctx.get(pid, name)
    return None


def _reference_transcriptions(s: Any) -> list:
    """The spoken-transcription GT from the bus (`reference_transcription`, published by
    the asr / audio_embedding node)."""
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


def _publish_retrieved(s: Any, results_list: list, query_ids: list) -> None:
    """Publish ``retrieved`` (keyed when ids align 1:1, else plain) via the shared publish
    contract — so rerank/metrics/answer_gen read the list, metric nodes read it keyed."""
    publish_keyed_or_plain(s, "retrieved", results_list, query_ids)
