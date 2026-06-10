"""Source stage handlers: dataset_source (graph root) + tts synthesis.

Moved verbatim from the former ``evaluation/phased.py`` (Phase 1, X5). Each handler
registers itself via ``@register_stage_handler`` at import time.
"""

from __future__ import annotations

import time
from typing import Any, List, Optional

from ..stage_registry import register_stage_handler
from ...logging_config import get_logger
from ..executor.state import RunState

logger = get_logger(__name__)


@register_stage_handler("dataset_source", self_timed=True)
def _stage_dataset_source(s: RunState) -> None:
    """Graph root: the dataset's source artifacts enter the DAG here. Loading + TTS
    happen in prepare_dataset before the graph (lifecycle reasons), so this validates
    the dataset is present + non-empty and logs its size; downstream nodes read it.

    Multi-dataset (B1): when the graph has several dataset_source nodes, each selects its source
    via ``params.dataset`` → ``s.dataset_sources[id]``; single-source falls back to ``s.dataset``.
    """
    dataset = _node_dataset(s)
    if dataset is None:
        raise ValueError("dataset_source: no dataset on the execution context")
    try:
        n = len(dataset)
    except TypeError:
        n = -1
    logger.info("dataset_source: %s samples", n if n >= 0 else "?")
    _publish_source_itemsets(s, dataset)


def _node_dataset(s: RunState) -> Any:
    """The dataset a node operates on (B1/B2): its ``params.dataset`` → ``s.dataset_sources[id]``,
    else the single ``s.dataset``. Lets ``corpus_index`` / ``dataset_source`` bind to a *specific*
    source in a multi-dataset graph; single-source graphs always get ``s.dataset``."""
    params = getattr(s.current_node, "params", None) or {}
    sid = params.get("dataset")
    if sid is not None:
        ds = s.dataset_sources.get(str(sid))
        if ds is not None:
            return ds
    return s.dataset


def _question_short_answer(question: Any) -> Optional[str]:
    """A reference short answer for a question, if the dataset carries one."""
    direct = getattr(question, "short_answer", None) or getattr(
        question, "answer", None
    )
    if direct:
        return str(direct)
    meta = getattr(question, "metadata", None) or {}
    for key in ("short_answer", "answer", "reference_answer"):
        if meta.get(key):
            return str(meta[key])
    return None


def _publish_source_itemsets(s: RunState, dataset: Any) -> None:
    """Publish per-item source artifacts as keyed ``ItemSet``s (architecture W1/A1/A3).

    Sourced cheaply from ``dataset.questions`` (BenchmarkQuestion metadata — no audio
    decode; order matches ``dataset[i]``). Additive: these ride alongside the legacy
    positional path and are consumed once metric nodes land (W4). No-op when the dataset
    exposes no ``questions`` (the positional path still covers those datasets).
    """
    questions = getattr(dataset, "questions", None)
    if not questions:
        return
    ids = [str(getattr(q, "question_id", i)) for i, q in enumerate(questions)]
    if len(set(ids)) != len(ids):
        logger.warning(
            "dataset_source: duplicate question_ids — skipping keyed ItemSet publish"
        )
        return

    from ..item_set import ItemSet

    s.put_artifact(
        "reference_text",
        ItemSet(ids, [getattr(q, "question_text", "") or "" for q in questions]),
    )

    relevance: List[dict] = []
    for q in questions:
        grades = getattr(q, "relevance_grades", None)
        if not grades:
            gt = getattr(q, "groundtruth_doc_ids", None) or []
            grades = {str(doc_id): 1 for doc_id in gt}
        relevance.append(dict(grades))
    s.put_artifact("relevant_docs", ItemSet(ids, relevance))

    answers = [_question_short_answer(q) for q in questions]
    if any(a is not None for a in answers):
        s.put_artifact("short_answers", ItemSet(ids, [a or "" for a in answers]))
    logger.debug("dataset_source: published %d keyed source items", len(ids))


@register_stage_handler("tts", self_timed=True)
def _stage_tts(s: RunState) -> None:
    """Synthesize query audio from the dataset's text (TTS bridge). Synthesizes only
    the genuinely-missing clips (cached / on-disk hits skipped); no-op when nothing is
    missing. No-op when config is absent (direct callers)."""
    if s.config is None:
        return
    questions = getattr(s.dataset, "questions", None)
    if not questions:
        return
    from ...pipeline.audio.prepare import synthesize_missing_query_audio

    _t = time.perf_counter()
    synthesize_missing_query_audio(
        questions,
        s.config.audio_synthesis,
        log=logger,
        cache_manager=s.cache_manager,
    )
    s.stage_times["tts_s"] = s.stage_times.get("tts_s", 0.0) + (
        time.perf_counter() - _t
    )
