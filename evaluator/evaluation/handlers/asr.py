"""ASR stage handler: produce query texts (or oracle bypass) for ASR-based modes.

Moved verbatim from the former ``evaluation/phased.py`` (Phase 1, X6). Registers the
``asr`` handler at import time.
"""

from __future__ import annotations

from ..stage_registry import register_stage_handler
from ...logging_config import get_logger
from ..helpers import _build_relevant_from_item
from ..executor.state import RunState
from ..executor.node_pipeline import _node_pipeline

logger = get_logger(__name__)


def _run_asr_phase(
    dataset,
    asr_pipeline,
    mode,
    oracle_mode,
    batch_size,
    num_workers,
    checkpoint_interval,
    experiment_id,
    resume_from_checkpoint,
):
    """Produce query texts for ASR-based modes.

    Returns (hypotheses, ground_truth, asr_hypotheses_for_wer, relevance, query_ids).
    In oracle mode, ground-truth transcriptions are used as queries (ASR skipped)
    so retrieval quality can be measured independently of ASR error. relevance and
    query_ids are populated for asr_text_retrieval (and oracle); empty otherwise.
    """
    if oracle_mode:
        # Oracle baseline: skip ASR, use ground-truth transcriptions directly.
        ground_truth = [
            str(dataset[i].get("transcription", dataset[i].get("question", "")))
            for i in range(len(dataset))
        ]
        hypotheses = list(ground_truth)
        asr_hyps_for_wer = list(hypotheses)
        relevance = [_build_relevant_from_item(dataset[i]) for i in range(len(dataset))]
        query_ids = [str(dataset[i].get("question_id", i)) for i in range(len(dataset))]
        logger.info(
            f"Oracle bypass complete: {len(hypotheses)} GT transcriptions used as queries"
        )
        return hypotheses, ground_truth, asr_hyps_for_wer, relevance, query_ids

    hypotheses, ground_truth = asr_pipeline.process_dataset(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        language=None,
        checkpoint_interval=checkpoint_interval,
        experiment_id=experiment_id,
        resume_from_checkpoint=resume_from_checkpoint,
    )
    logger.info(f"ASR Phase complete: {len(hypotheses)} transcriptions")
    # Snapshot raw ASR output — WER/CER must compare against this, not any
    # query-optimized version that may expand the text significantly.
    asr_hyps_for_wer = list(hypotheses)

    relevance: list = []
    query_ids: list = []
    # Query ids align the per-item ASR scores to keyed ItemSets so the metric registry can
    # score WER/CER (L3a) — needed for asr_only too, which has no retrieval/relevance.
    if mode in ("asr_text_retrieval", "asr_only"):
        query_ids = [str(dataset[i].get("question_id", i)) for i in range(len(dataset))]
    if mode == "asr_text_retrieval":
        # Keep relevance aligned with query order for IR metrics.
        relevance = [_build_relevant_from_item(dataset[i]) for i in range(len(dataset))]
    return hypotheses, ground_truth, asr_hyps_for_wer, relevance, query_ids


@register_stage_handler("asr", time_key="asr_s")
def _stage_asr(s: RunState) -> None:
    """ASR (or oracle bypass): produce query texts + relevance for ASR modes.

    Per-branch divergence (R2): a node ``params.oracle: true`` makes *this* branch use the
    reference transcriptions (the oracle/ref branch) even when the run isn't globally oracle.
    """
    params = getattr(s.current_node, "params", None) or {}
    oracle = bool(s.oracle_mode or params.get("oracle"))
    s.cb(
        "phase_1_asr",
        0,
        s.total,
        "Phase 1: Oracle bypass" if oracle else "Phase 1: ASR transcription",
    )
    with _node_pipeline(s, "asr", params):
        (
            s.all_hypotheses,
            s.all_ground_truth,
            asr_hypotheses_for_wer,
            s.all_relevance,
            s.all_query_ids,
        ) = _run_asr_phase(
            s.dataset,
            s.asr_pipeline,
            s.mode,
            oracle,
            s.batch_size,
            s.num_workers,
            s.checkpoint_interval,
            s.experiment_id,
            s.resume_from_checkpoint,
        )
    _publish_query_text(s, s.all_hypotheses)  # ASR hypotheses feed embedding (R4d)
    # Also publish the RAW ASR output (pre-correction) as a keyed artifact so the registry
    # can score raw_wer/raw_cer against it even when a downstream query_correction node
    # republishes a corrected `query_text` (L1). This is the bus copy of the raw snapshot —
    # the metrics node reads it from here (no RunState mirror, T4). Keyed only when ids align.
    _ids = [str(i) for i in (s.all_query_ids or [])]
    if len(_ids) == len(asr_hypotheses_for_wer) and len(set(_ids)) == len(_ids):
        from ..item_set import ItemSet

        s.put_items("raw_query_text", ItemSet(_ids, list(asr_hypotheses_for_wer)))
    else:
        # Non-aligned ids (no keying): still publish the raw snapshot as a plain list so the
        # metrics node reads raw ASR text from the bus rather than a RunState mirror (T4).
        s.put_artifact("raw_query_text", list(asr_hypotheses_for_wer))
    s.cb(
        "phase_1_asr",
        s.total,
        s.total,
        "Oracle bypass complete" if s.oracle_mode else "ASR transcription complete",
    )


def _publish_query_text(s: RunState, texts: list) -> None:
    """Publish ``query_text`` as a keyed ``ItemSet`` when query ids align 1:1 (W2);
    otherwise the plain list (legacy path, e.g. asr_only with no query ids). Either way
    ``get_artifact('query_text')`` returns the list, so embedding/optimization are unchanged.
    """
    ids = [str(i) for i in (s.all_query_ids or [])]
    if len(ids) == len(texts) and len(set(ids)) == len(ids):
        from ..item_set import ItemSet

        s.put_items("query_text", ItemSet(ids, list(texts)))
    else:
        s.put_artifact("query_text", texts)
