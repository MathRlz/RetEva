"""ASR stage handler: produce query texts (or oracle bypass) for ASR-based modes.

Moved verbatim from the former ``evaluation/phased.py`` (Phase 1, X6). Registers the
``asr`` handler at import time.
"""

from __future__ import annotations

from ..stage_registry import register_stage_handler
from ._common import publish_keyed_or_plain as _publish_keyed_or_plain  # noqa: F401
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
    warmup_batch_sizing=False,
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
        warmup_batch_sizing=warmup_batch_sizing,
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


@register_stage_handler("convert", time_key="asr_s")
def _stage_convert(s: RunState) -> None:
    """The ``convert`` operator (modality change): dispatch by op to ASR (audio→text) or
    TTS (text→audio). Bodies unchanged; tts lives in the audio handlers."""
    from .audio import _stage_tts
    from ._dispatch import dispatch_operator

    return dispatch_operator("convert", {
        "asr": _stage_asr,
        "tts": _stage_tts,
    }, s)


def _stage_asr(s: RunState) -> None:
    """ASR (or oracle bypass): produce query texts + relevance for ASR modes.

    Per-branch divergence (R2): a node ``params.oracle: true`` makes *this* branch use the
    reference transcriptions (the oracle/ref branch) even when the run isn't globally oracle.
    """
    params = s.node_params
    oracle = bool(s.oracle_mode or params.get("oracle"))
    s.cb(
        "phase_1_asr",
        0,
        s.total,
        "Phase 1: Oracle bypass" if oracle else "Phase 1: ASR transcription",
    )
    # Bus-first audio (P4): a republished query_audio ref set (augment_audio /
    # union) wraps the dataset in a ref-decoding view; matching refs pass through.
    from ..audio_refs import resolve_audio_dataset

    audio_dataset = resolve_audio_dataset(s, s.dataset)
    with _node_pipeline(s, "asr", params):
        (
            hypotheses,
            ground_truth,
            asr_hypotheses_for_wer,
            relevance,
            query_ids,
        ) = _run_asr_phase(
            audio_dataset,
            s.asr_pipeline,
            s.mode,
            oracle,
            s.batch_size,
            s.num_workers,
            s.checkpoint_interval,
            s.experiment_id,
            s.resume_from_checkpoint,
            warmup_batch_sizing=bool(
                getattr(getattr(s.config, "data", None), "warmup_batch_sizing", False)
            ),
        )
    # A checkpoint resume can leave a longer, batch-overlapping hypotheses list (the
    # legacy zip-truncation leniency); trim to the dataset's query ids so the keyed
    # publish stays aligned (M1d-2 — was handled by the metrics-side trim before).
    if query_ids and len(hypotheses) > len(query_ids):
        logger.debug(
            "asr: trimming %d hypotheses to %d query ids (checkpoint overlap)",
            len(hypotheses),
            len(query_ids),
        )
        hypotheses = hypotheses[: len(query_ids)]
    # Bus-only handoff (M1d-2): everything downstream reads keyed ctx artifacts.
    # Pure transform: asr publishes ONLY its hypothesis. It is never overwritten
    # (correction/optimization emit distinct names), so query_text stays the un-rewritten
    # ASR output WER/CER score against — no raw_query_text needed. Ground truth comes from
    # dataset_source.
    _publish_keyed_or_plain(s, "query_text", hypotheses, query_ids)  # feeds embedding (R4d)
    s.cb(
        "phase_1_asr",
        s.total,
        s.total,
        "Oracle bypass complete" if s.oracle_mode else "ASR transcription complete",
    )


