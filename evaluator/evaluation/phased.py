"""Phased evaluation for efficient batch processing.

This module contains the evaluate_phased() function which processes the entire
dataset in sequential phases (ASR/Embedding, Text Embedding, Retrieval, Metrics)
for better GPU utilization and throughput.
"""
from typing import Callable, Optional, Dict, Any, List
from dataclasses import dataclass, field
import time
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from ..datasets import QueryDataset
from ..pipeline import (
    ASRPipelineProtocol,
    TextEmbeddingPipelineProtocol,
    AudioEmbeddingPipelineProtocol,
    RetrievalPipelineProtocol,
    build_stage_graph,
)
from .stage_registry import register_stage_handler, get_stage_spec
from ..storage.cache import CacheManager
from ..logging_config import get_logger, TimingContext, log_cache_stats
from ..judge import run_llm_judging
from ..metrics import (
    compute_ir_metrics,
    log_ir_metrics,
    first_relevant_rank_distribution,
    wer_recall_correlation,
    categorize_failures,
    embedding_alignment,
    per_speaker_breakdown,
    judge_calibration,
)

from .helpers import (
    _payload_to_key,
    _search_results_to_keys,
    _build_relevant_from_item,
    collate_fn,
    detect_pipeline_mode,
    PIPELINE_MODE_LABELS,
)
from .metrics import word_error_rate, character_error_rate
from ..metrics.stt import term_weighted_wer
from ..metrics.ir import recall_at_k, reciprocal_rank, ndcg_at_k
from ..metrics.domain_terms import load_term_weights
from ..analysis.errors import analyze_retrieval_failures
from ..analysis.significance import bootstrap_confidence_interval
from ..models.retrieval.embedding_fusion import fuse_embeddings, validate_fusion_config
from ..models.retrieval.query.optimization import (
    rewrite_query,
    generate_hypothetical_document,
    decompose_query,
    generate_multi_queries,
    combine_retrieval_results,
)
from .answer_gen import generate_answers
from .result_schema import PhasedResults

logger = get_logger(__name__)

# Phase banner + retrieval-debug formatting.
LOG_DIVIDER = "=" * 50
DEBUG_SAMPLE_LIMIT = 3
MATCH_SYMBOL = "✓"
MISS_SYMBOL = "✗"
# Bootstrap confidence-interval settings.
BOOTSTRAP_ALPHA = 0.05
BOOTSTRAP_ITERATIONS = 1000
MIN_SAMPLES_FOR_CI = 20


def _log_phase(title: str) -> None:
    """Log a phase banner: divider, title, divider."""
    logger.info(LOG_DIVIDER)
    logger.info(title)
    logger.info(LOG_DIVIDER)


@dataclass
class EvaluationContext:
    """Bundles the pipeline + execution parameters of ``evaluate_phased``.

    Pass an instance as ``evaluate_phased(dataset, context=ctx)`` (or
    ``evaluate_phased(dataset, context=ctx, ...)``) instead of threading the dozen
    individual keyword arguments. When a context is given it supersedes the
    individual kwargs; omit it to keep using the explicit kwargs. ``features``
    carries the optional, default-off feature configs (see ``PhasedFeatures``).
    """
    # Pipelines
    retrieval_pipeline: Optional[RetrievalPipelineProtocol] = None
    asr_pipeline: Optional[ASRPipelineProtocol] = None
    text_embedding_pipeline: Optional[TextEmbeddingPipelineProtocol] = None
    audio_embedding_pipeline: Optional[AudioEmbeddingPipelineProtocol] = None
    
    # Execution parameters
    k: int = 10
    batch_size: int = 32
    trace_limit: int = 0
    num_workers: int = 0
    checkpoint_interval: int = 500
    experiment_id: Optional[str] = None
    resume_from_checkpoint: bool = True
    progress_callback: Optional[Callable[[str, int, int, str], None]] = None
    oracle_mode: bool = False
    
    # Cache and features
    cache_manager: Optional[CacheManager] = None
    features: Optional["PhasedFeatures"] = None


def _log_embedding_stats(embeddings, label: str) -> None:
    """Log shape / mean / std / norm of an embedding matrix (debug aid)."""
    if len(embeddings) == 0:
        return
    emb_array = np.array(embeddings)
    logger.info(f"{label} shape: {emb_array.shape}")
    logger.info(f"{label} stats - mean: {emb_array.mean():.4f}, std: {emb_array.std():.4f}")
    logger.info(f"{label} norms - mean: {np.linalg.norm(emb_array, axis=1).mean():.4f}")


def _embed_all_audio(dataset, audio_embedding_pipeline, batch_size, num_workers):
    """Run the audio-embedding pass over the whole dataset.

    Returns parallel lists (embeddings, ground_truth, relevance, query_ids).
    Shared by audio_text_retrieval and audio_emb_retrieval modes.
    """
    embeddings: list = []
    ground_truth: list = []
    relevance: list = []
    query_ids: list = []
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    with TimingContext("Audio Embedding Phase", logger):
        for batch in tqdm(dataloader, desc="Audio embedding"):
            audio_list = [item["audio_array"] for item in batch]
            sampling_rates = [item["sampling_rate"] for item in batch]
            transcriptions = [item["transcription"] for item in batch]
            relevance_batch = [_build_relevant_from_item(item) for item in batch]
            query_ids_batch = [
                str(item.get("question_id", len(query_ids) + idx))
                for idx, item in enumerate(batch)
            ]
            batch_embeddings = audio_embedding_pipeline.process_batch(audio_list, sampling_rates)
            embeddings.extend(batch_embeddings)
            ground_truth.extend(transcriptions)
            relevance.extend(relevance_batch)
            query_ids.extend(query_ids_batch)
    return embeddings, ground_truth, relevance, query_ids


def _run_query_optimization_phase(
    mode,
    all_hypotheses,
    all_ground_truth,
    all_results_with_scores,
    query_opt_config,
    text_embedding_pipeline,
    retrieval_pipeline,
    k,
):
    """Apply query optimization to the per-query texts before retrieval.

    rewrite/hyde transform the query text in place. decompose/multi_query expand
    each query, retrieve per sub-query, and merge — fully bypassing the normal
    embedding + retrieval phases.

    Returns (all_hypotheses, all_ground_truth, all_results_with_scores,
    all_retrieved, bypassed). all_retrieved is a list only when bypassed, else None.
    """
    method = query_opt_config.method
    query_texts_src = all_hypotheses if mode == "asr_text_retrieval" else all_ground_truth
    all_retrieved = None
    bypassed = False

    if method in ("rewrite", "hyde"):
        fn = rewrite_query if method == "rewrite" else generate_hypothetical_document
        optimized = []
        # NOTE: rewrite_query supports context-aware iterative refinement via its
        # `context=` arg (config.use_initial_context / context_top_k), but this single
        # pass calls it without context — that capability is not wired here yet.
        # Wiring it would require an initial retrieval to feed top-k docs back in.
        for q in tqdm(query_texts_src, desc=f"Query optimization ({method})"):
            try:
                optimized.append(fn(q, query_opt_config))
            except Exception as exc:
                logger.warning("Query optimization failed for %r: %s", q[:80], exc)
                optimized.append(q)
        if mode == "asr_text_retrieval":
            all_hypotheses = optimized
        else:
            all_ground_truth = optimized
        logger.info("Query optimization complete: %d queries transformed", len(optimized))

    elif method in ("decompose", "multi_query"):
        if text_embedding_pipeline is None or retrieval_pipeline is None:
            logger.warning(
                "Query optimization %s requires text_embedding + retrieval pipeline — skipping",
                method,
            )
        else:
            fn = decompose_query if method == "decompose" else generate_multi_queries
            all_results_with_scores = []
            all_retrieved = []
            for q in tqdm(query_texts_src, desc=f"Query optimization ({method})"):
                try:
                    sub_qs = fn(q, query_opt_config)
                except Exception as exc:
                    logger.warning("Query expansion failed for %r: %s", q[:80], exc)
                    sub_qs = [q]
                sub_embs = np.array(text_embedding_pipeline.process_batch(sub_qs))
                sub_results = retrieval_pipeline.search_batch(sub_embs, k, query_texts=sub_qs)
                merged = combine_retrieval_results(
                    sub_results,
                    strategy=query_opt_config.combine_strategy,
                    k=k,
                )
                all_results_with_scores.append(merged)
                all_retrieved.append(_search_results_to_keys(merged))
            bypassed = True
            logger.info(
                "Query optimization (%s) complete: %d queries expanded and retrieved",
                method, len(all_results_with_scores),
            )

    return all_hypotheses, all_ground_truth, all_results_with_scores, all_retrieved, bypassed


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
        _log_phase("PHASE 1: Oracle bypass (GT transcriptions as queries)")
        ground_truth = [
            str(dataset[i].get("transcription", dataset[i].get("question", "")))
            for i in range(len(dataset))
        ]
        hypotheses = list(ground_truth)
        asr_hyps_for_wer = list(hypotheses)
        relevance = [_build_relevant_from_item(dataset[i]) for i in range(len(dataset))]
        query_ids = [str(dataset[i].get("question_id", i)) for i in range(len(dataset))]
        logger.info(f"Oracle bypass complete: {len(hypotheses)} GT transcriptions used as queries")
        return hypotheses, ground_truth, asr_hyps_for_wer, relevance, query_ids

    _log_phase("PHASE 1: ASR Transcription")
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
    if mode == "asr_text_retrieval":
        # Keep relevance aligned with query order for IR metrics.
        relevance = [_build_relevant_from_item(dataset[i]) for i in range(len(dataset))]
        query_ids = [str(dataset[i].get("question_id", i)) for i in range(len(dataset))]
    return hypotheses, ground_truth, asr_hyps_for_wer, relevance, query_ids


def _log_retrieval_debug(
    retrieval_pipeline,
    all_retrieved,
    all_ground_truth,
    all_relevance,
    all_hypotheses,
    all_query_ids,
    all_embeddings,
    mode,
    k,
) -> None:
    """Verbose per-query retrieval inspection (dense + no-rerank only).

    Pure diagnostic logging extracted from the retrieval phase; has no effect on
    results. Logs DB vector stats and a re-search trace for the first 3 queries.
    """
    # Debug: Check vector store properties
    if hasattr(retrieval_pipeline.vector_store, 'vectors') and retrieval_pipeline.vector_store.vectors is not None:
        db_vectors = retrieval_pipeline.vector_store.vectors
        logger.info(f"DB vectors shape: {db_vectors.shape}")
        logger.info(f"DB vectors stats - mean: {db_vectors.mean():.4f}, std: {db_vectors.std():.4f}")
        logger.info(f"DB vectors norms - mean: {np.linalg.norm(db_vectors, axis=1).mean():.4f}")
        logger.info(f"DB payload count: {len(getattr(retrieval_pipeline.vector_store, 'payloads', []))}")

    if not (len(all_retrieved) > 0 and len(all_ground_truth) > 0):
        return

    can_debug_with_search = (
        retrieval_pipeline.strategy_config.core.mode == "dense"
        and retrieval_pipeline.strategy_config.reranking.mode == "none"
    )
    if not can_debug_with_search:
        logger.info("Skipping dense-only debug re-search for current retrieval mode/reranker")
        return

    _log_phase(f"RETRIEVAL DEBUG SAMPLE (first {DEBUG_SAMPLE_LIMIT} queries):")

    # Build a doc_id → text lookup from vector store payloads for ground-truth display
    _db_payloads = getattr(retrieval_pipeline.vector_store, 'payloads', [])
    _doc_text_lookup: dict = {}
    for _p in _db_payloads:
        if isinstance(_p, dict):
            _did = str(_p.get("doc_id", ""))
            _dtxt = _p.get("text") or _p.get("abstract") or _p.get("title") or ""
            if _did:
                _doc_text_lookup[_did] = str(_dtxt)

    emb_array = np.array(all_embeddings) if isinstance(all_embeddings, list) else all_embeddings
    for i in range(min(DEBUG_SAMPLE_LIMIT, len(all_embeddings))):
        results_with_scores = retrieval_pipeline.search(emb_array[i], k=10)

        # gt_doc_id: the expected document id
        if i < len(all_relevance) and len(all_relevance[i]) > 0:
            gt_doc_id = next(iter(all_relevance[i].keys()))
        else:
            gt_doc_id = str(all_ground_truth[i])

        # query_text: what was actually searched (ASR output or reference)
        if mode == "asr_text_retrieval" and i < len(all_hypotheses):
            query_text = all_hypotheses[i]
        elif i < len(all_ground_truth):
            query_text = str(all_ground_truth[i])
        else:
            query_text = all_query_ids[i] if i < len(all_query_ids) else "?"

        # gt_doc_text: the text of the expected document (from corpus)
        gt_doc_text = _doc_text_lookup.get(gt_doc_id, "")

        logger.info(f"\nQuery {i+1}:")
        logger.info(f"  ASR query text:        '{query_text[:120]}'")
        if mode == "asr_text_retrieval" and i < len(all_ground_truth):
            ref = str(all_ground_truth[i])
            if ref != query_text:
                logger.info(f"  Ground truth question: '{ref[:120]}'")
        logger.info(f"  Expected doc:  [{gt_doc_id}] '{gt_doc_text[:100]}'")
        logger.info(f"  Query emb norm: {np.linalg.norm(emb_array[i]):.4f}")
        logger.info(f"  Top 5 retrieved:")

        for j, (payload, score) in enumerate(results_with_scores[:5], 1):
            doc_id = payload.get("doc_id", "") if isinstance(payload, dict) else str(payload)
            match = MATCH_SYMBOL if doc_id == gt_doc_id else MISS_SYMBOL
            preview = (payload.get("text", str(payload)) if isinstance(payload, dict) else str(payload))[:80]
            logger.info(f"    {j}. [{match}] Score: {score:.4f} | [{doc_id}] '{preview}'")

        gt_rank = None
        for rank, (payload, score) in enumerate(results_with_scores, 1):
            doc_id = payload.get("doc_id", "") if isinstance(payload, dict) else str(payload)
            if doc_id == gt_doc_id:
                gt_rank = rank
                logger.info(f"  Expected doc found at rank: {gt_rank} (score: {score:.4f})")
                break

        if gt_rank is None:
            logger.info(f"  Expected doc NOT found in top-{k}")
            payload_keys = [_payload_to_key(p) for p in _db_payloads]
            if gt_doc_id in payload_keys:
                logger.info(f"  (Doc exists in DB but outside top-{k})")
            else:
                logger.info(f"  (Doc NOT in DB at all!)")


@dataclass
class PhasedFeatures:
    """Optional feature configs + analysis flags for ``evaluate_phased``.

    Groups the rarely-set, default-off knobs so the engine signature stays focused
    on the runtime inputs (pipelines, dataset, batch params). All default to
    disabled; ``evaluate_from_bundle`` builds one from an ``EvaluationConfig``.
    """
    judge_config: Any = None
    answer_gen_config: Any = None
    query_opt_config: Any = None
    embedding_fusion_config: Any = None
    term_weights: Optional[Dict[str, float]] = None
    compute_confidence_intervals: bool = False


@dataclass
class _PhasedState:
    """Mutable execution context threaded through DAG stage handlers.

    Holds the pipelines/config inputs plus the accumulators each stage reads and
    writes. One instance per evaluate_phased call; handlers mutate it in place.
    """
    # inputs
    dataset: Any
    mode: str
    retrieval_pipeline: Any
    asr_pipeline: Any
    text_embedding_pipeline: Any
    audio_embedding_pipeline: Any
    cache_manager: Any
    k: int
    batch_size: int
    num_workers: int
    checkpoint_interval: int
    experiment_id: Any
    resume_from_checkpoint: bool
    oracle_mode: bool
    embedding_fusion_config: Any
    query_opt_config: Any
    answer_gen_config: Any
    judge_config: Any
    trace_limit: int
    term_weights: Any
    compute_confidence_intervals: bool
    total: int
    cb: Callable
    t_total: float = 0.0
    # Model lifecycle: when set, a stage's model is released after the last stage that
    # uses it (frees the device mid-run). Off unless a provider + on_finish policy apply.
    service_provider: Any = None
    offload_after_stage: bool = False
    # accumulators (parallel per-query lists unless noted)
    all_hypotheses: list = field(default_factory=list)
    all_ground_truth: list = field(default_factory=list)
    all_embeddings: Any = field(default_factory=list)   # retrieval input
    audio_embeddings: Any = None                          # raw audio emb (for fusion)
    text_embeddings_for_fusion: Any = None
    all_relevance: list = field(default_factory=list)
    asr_hypotheses_for_wer: list = field(default_factory=list)
    all_query_ids: list = field(default_factory=list)
    all_results_with_scores: list = field(default_factory=list)
    all_retrieved: list = field(default_factory=list)
    audio_emb_for_alignment: Any = None
    text_emb_for_alignment: Any = None
    query_opt_bypassed: bool = False
    stage_times: Dict[str, float] = field(default_factory=dict)
    results: "PhasedResults" = field(default_factory=lambda: PhasedResults())
    # Intermediates the metrics stage hands to the answer-gen / finalize stages.
    metrics_all_relevant: list = field(default_factory=list)
    wer_scores: list = field(default_factory=list)
    cer_scores: list = field(default_factory=list)
    per_query_recall5: list = field(default_factory=list)
    ans_detail_by_qid: dict = field(default_factory=dict)


@register_stage_handler("asr", time_key="asr_s")
def _stage_asr(s: _PhasedState) -> None:
    """ASR (or oracle bypass): produce query texts + relevance for ASR modes."""
    s.cb("phase_1_asr", 0, s.total, "Phase 1: Oracle bypass" if s.oracle_mode else "Phase 1: ASR transcription")
    (
        s.all_hypotheses,
        s.all_ground_truth,
        s.asr_hypotheses_for_wer,
        s.all_relevance,
        s.all_query_ids,
    ) = _run_asr_phase(
        s.dataset, s.asr_pipeline, s.mode, s.oracle_mode, s.batch_size, s.num_workers,
        s.checkpoint_interval, s.experiment_id, s.resume_from_checkpoint,
    )
    s.cb("phase_1_asr", s.total, s.total, "Oracle bypass complete" if s.oracle_mode else "ASR transcription complete")


@register_stage_handler("audio_embedding", time_key="asr_s")
def _stage_audio_embedding(s: _PhasedState) -> None:
    """Embed all audio; becomes the retrieval input unless fusion overrides it."""
    _log_phase("PHASE 1: Audio Embedding")
    s.cb("phase_1_audio", 0, s.total, "Phase 1: Audio embedding")
    emb, gt, rel, qids = _embed_all_audio(
        s.dataset, s.audio_embedding_pipeline, s.batch_size, s.num_workers
    )
    s.all_embeddings = emb
    s.audio_embeddings = np.array(emb) if len(emb) > 0 else emb
    s.all_ground_truth, s.all_relevance, s.all_query_ids = gt, rel, qids
    logger.info(f"Audio Embedding Phase complete: {len(s.all_embeddings)} audio embeddings")
    s.cb("phase_1_audio", s.total, s.total, "Audio embedding complete")
    _log_embedding_stats(s.all_embeddings, "Audio embedding")


@register_stage_handler("text_embedding", time_key="embedding_s")
def _stage_text_embedding(s: _PhasedState) -> None:
    """Text embedding. For ASR modes embeds the (optimized) hypotheses into the
    retrieval input; for fused audio_text it embeds reference texts for fusion."""
    if s.mode == "asr_text_retrieval":
        if s.query_opt_bypassed:
            return
        _log_phase("PHASE 2: Text Embedding (from ASR)")
        s.cb("phase_2_embedding", 0, s.total, "Phase 2: Text embedding")
        with TimingContext("Embedding Phase", logger):
            s.all_embeddings = s.text_embedding_pipeline.process_batch(
                s.all_hypotheses, show_progress=True, desc="Embedding query transcriptions (ASR)",
            )
        logger.info(f"Text Embedding Phase complete: {len(s.all_embeddings)} embeddings")
        s.cb("phase_2_embedding", s.total, s.total, "Text embedding complete")
        if s.cache_manager and s.experiment_id:
            s.cache_manager.set_checkpoint(
                f"{s.experiment_id}_phased",
                {"phase": "embedding", "hypotheses": s.all_hypotheses, "ground_truth": s.all_ground_truth},
            )
    elif s.mode == "audio_text_retrieval":
        # Fusion path: embed reference texts (skipped gracefully if no pipeline).
        if s.text_embedding_pipeline is None:
            return
        _log_phase("PHASE 1.5: Text Embedding & Fusion")
        s.cb("phase_1_5_fusion", 0, s.total, "Phase 1.5: Text embedding & fusion")
        with TimingContext("Text Embedding for Fusion", logger):
            te = s.text_embedding_pipeline.process_batch(
                s.all_ground_truth, show_progress=True, desc="Embedding reference texts (fusion)",
            )
            s.text_embeddings_for_fusion = np.array(te)
        logger.info(f"Text embedding shape: {s.text_embeddings_for_fusion.shape}")
        logger.info(
            f"Text embedding stats - mean: {s.text_embeddings_for_fusion.mean():.4f}, "
            f"std: {s.text_embeddings_for_fusion.std():.4f}"
        )


@register_stage_handler("fusion", time_key="embedding_s")
def _stage_fusion(s: _PhasedState) -> None:
    """Fuse audio + reference-text embeddings into the retrieval input."""
    if s.text_embedding_pipeline is None or s.text_embeddings_for_fusion is None:
        logger.warning(
            "Embedding fusion is enabled but text_embedding_pipeline is not available. "
            "Skipping fusion and using audio embeddings only."
        )
        return
    audio_arr = s.audio_embeddings
    text_arr = s.text_embeddings_for_fusion
    s.audio_emb_for_alignment = audio_arr
    s.text_emb_for_alignment = text_arr
    validate_fusion_config(
        s.embedding_fusion_config, audio_dim=audio_arr.shape[1], text_dim=text_arr.shape[1]
    )
    with TimingContext("Embedding Fusion", logger):
        fused_embeddings, fusion_meta = fuse_embeddings(
            audio_arr, text_arr, s.embedding_fusion_config
        )
    if fusion_meta:
        logger.debug("Fusion metadata: %s", fusion_meta)
    logger.info(f"Fused embedding shape: {fused_embeddings.shape}")
    logger.info(f"Fused embedding stats - mean: {fused_embeddings.mean():.4f}, std: {fused_embeddings.std():.4f}")
    logger.info(f"Fused embedding norms - mean: {np.linalg.norm(fused_embeddings, axis=1).mean():.4f}")
    logger.info(f"Using fused embeddings for retrieval (method: {s.embedding_fusion_config.fusion_method})")
    s.all_embeddings = fused_embeddings


@register_stage_handler("retrieval", time_key="retrieval_s")
def _stage_retrieval(s: _PhasedState) -> None:
    """Vector search over the retrieval-input embeddings."""
    if s.query_opt_bypassed:
        return
    if isinstance(s.all_embeddings, list) and len(s.all_embeddings) > 0:
        s.all_embeddings = np.array(s.all_embeddings)
    _log_phase("PHASE 3: Retrieval")
    s.cb("phase_3_retrieval", 0, s.total, "Phase 3: Retrieval")
    with TimingContext("Retrieval Phase", logger):
        query_texts = None
        if s.mode == "asr_text_retrieval":
            query_texts = s.all_hypotheses
        elif s.mode == "audio_text_retrieval":
            query_texts = s.all_ground_truth
        elif s.mode == "audio_emb_retrieval":
            if s.retrieval_pipeline.strategy_config.core.mode != "dense":
                raise ValueError(
                    "audio_emb_retrieval supports only retrieval_mode='dense'. "
                    "Sparse/hybrid requires query text unavailable in this path."
                )
        results_list = s.retrieval_pipeline.search_batch(s.all_embeddings, s.k, query_texts=query_texts)
        s.all_results_with_scores = results_list
        s.all_retrieved = [_search_results_to_keys(results) for results in results_list]
    logger.info(f"Retrieval Phase complete: {len(s.all_retrieved)} queries")
    s.cb("phase_3_retrieval", s.total, s.total, "Retrieval complete")
    _log_retrieval_debug(
        s.retrieval_pipeline, s.all_retrieved, s.all_ground_truth, s.all_relevance,
        s.all_hypotheses, s.all_query_ids, s.all_embeddings, s.mode, s.k,
    )


def _generate_answer_details(s: "_PhasedState", results, all_relevant) -> dict:
    """PHASE 5: run answer generation (if enabled); return query_id → detail map."""
    cfg = s.answer_gen_config
    if cfg is None or not getattr(cfg, "enabled", False):
        return {}

    _log_phase("PHASE 5: Answer Generation")
    s.cb("phase_5_answer_gen", 0, s.total, "Phase 5: Answer generation")
    query_texts = s.all_hypotheses if s.mode == "asr_text_retrieval" else s.all_ground_truth

    # Build corpus_lookup from the full corpus so reference answers can be found
    # even for retrieval misses (not just retrieved docs).
    corpus_lookup: dict = {}
    if hasattr(s.dataset, "get_corpus"):
        for doc in s.dataset.get_corpus():
            corpus_lookup[str(doc.get("doc_id", ""))] = doc
    # Overlay with retrieved payloads (may carry richer runtime metadata).
    for idx, result in enumerate(s.all_results_with_scores):
        for payload, _ in result:
            corpus_lookup[str(payload.get("doc_id", payload.get("id", idx)))] = payload

    answer_results = generate_answers(
        traces_data=(s.all_query_ids, all_relevant, s.all_results_with_scores),
        all_query_texts=query_texts,
        corpus_lookup=corpus_lookup,
        config=cfg,
    )
    results["answer_generation"] = answer_results
    logger.info(
        "Answer generation complete — %d cases, mean ROUGE-L: %s",
        answer_results["cases"],
        f"{answer_results['mean_rougeL']:.4f}" if answer_results["mean_rougeL"] is not None else "n/a",
    )
    return {d["query_id"]: d for d in answer_results["details"]}


def _build_query_traces(
    s: "_PhasedState", results, all_relevant, wer_scores, cer_scores,
    per_query_recall5, ans_detail_by_qid,
) -> None:
    """Build per-query traces (+ per-speaker breakdown) when trace_limit > 0."""
    if not (s.trace_limit > 0 and s.all_results_with_scores):
        return

    limit = min(s.trace_limit, len(s.all_results_with_scores))
    traces = []
    for i in range(limit):
        retrieved = [
            {"doc_key": _payload_to_key(payload), "score": float(score)}
            for payload, score in s.all_results_with_scores[i]
        ]
        query_id = s.all_query_ids[i] if i < len(s.all_query_ids) else str(i)
        ans_detail = ans_detail_by_qid.get(query_id, {})
        sample_meta = s.dataset[i].get("metadata", {}) if i < len(s.dataset) else {}
        trace_entry: Dict[str, Any] = {
            "query_id": query_id,
            "relevant": all_relevant[i] if i < len(all_relevant) else {},
            "retrieved": retrieved,
            "question": (s.all_ground_truth[i] if i < len(s.all_ground_truth) else ans_detail.get("question", "")),
            "generated_answer": ans_detail.get("generated_answer", ""),
            "reference_answer": ans_detail.get("reference_answer", ""),
            "metadata": sample_meta,
        }
        if i < len(wer_scores):
            trace_entry["per_query_wer"] = wer_scores[i]
            trace_entry["per_query_cer"] = cer_scores[i]
        if i < len(per_query_recall5):
            trace_entry["recall_at_5"] = per_query_recall5[i]
        traces.append(trace_entry)
    results["query_traces"] = traces

    per_speaker = per_speaker_breakdown(traces)
    if per_speaker is not None:
        results["per_speaker"] = per_speaker


def _run_judge(s: "_PhasedState", results, all_relevant, per_query_recall5) -> None:
    """PHASE 6: run the LLM judge over query traces (if enabled) + calibration."""
    cfg = s.judge_config
    if cfg is None or not getattr(cfg, "enabled", False):
        return
    if "query_traces" not in results:
        raise RuntimeError("LLM judge requires query traces; set trace_limit > 0")

    _log_phase("PHASE 6: Judge")
    s.cb("phase_6_judge", 0, s.total, "Phase 6: LLM judge")
    judge_mode = getattr(cfg, "judge_mode", "retrieval")
    logger.info(
        "Running LLM judge — mode=%s model=%s cases=%d",
        judge_mode,
        getattr(cfg, "model", "unknown"),
        getattr(cfg, "max_cases", -1),
    )
    judge_results = run_llm_judging(results["query_traces"], cfg, judge_mode=judge_mode)
    results["llm_judge"] = judge_results

    # Judge calibration: correlate per-query judge score with IR metrics.
    judge_scores = [
        d.get("judge", {}).get("score", float("nan"))
        for d in judge_results.get("details", [])
    ]
    calibration = judge_calibration(
        judge_scores, per_query_recall5, s.all_retrieved, all_relevant
    )
    if calibration:
        results.update(calibration)
        logger.info(
            "Judge calibration — vs_MRR=%.3f vs_Recall5=%.3f",
            results.get("judge_vs_MRR_correlation", float("nan")),
            results.get("judge_vs_Recall5_correlation", float("nan")),
        )

    logger.info(
        "Judge complete — cases=%d mean_score=%.4f pass_rate=%.1f%%",
        judge_results["cases"],
        judge_results["mean_score"],
        100.0 * sum(
            1 for r in judge_results["details"]
            if r.get("judge", {}).get("verdict") == "PASS"
        ) / max(len(judge_results["details"]), 1),
    )


@register_stage_handler("metrics", self_timed=True)
def _stage_metrics(s: _PhasedState) -> None:
    """Terminal node: compute ASR/IR metrics, traces, answer-gen, judge → results."""
    # Bind state into locals so the metric logic reads naturally.
    mode = s.mode
    dataset = s.dataset
    k = s.k
    _total = s.total
    _cb = s.cb
    asr_pipeline = s.asr_pipeline
    text_embedding_pipeline = s.text_embedding_pipeline
    audio_embedding_pipeline = s.audio_embedding_pipeline
    all_hypotheses = s.all_hypotheses
    all_ground_truth = s.all_ground_truth
    all_relevance = s.all_relevance
    all_retrieved = s.all_retrieved
    _asr_hypotheses_for_wer = s.asr_hypotheses_for_wer
    _audio_emb_for_alignment = s.audio_emb_for_alignment
    _text_emb_for_alignment = s.text_emb_for_alignment
    term_weights = s.term_weights
    trace_limit = s.trace_limit
    compute_confidence_intervals = s.compute_confidence_intervals
    _stage_times = s.stage_times
    _t_phase = time.perf_counter()

    _log_phase("PHASE 4: IR Metrics")
    _cb("phase_4_metrics", 0, _total, "Phase 4: Computing IR metrics")

    results: PhasedResults = PhasedResults()
    results["pipeline_mode"] = mode
    results["phased"] = True
    results["oracle_mode"] = s.oracle_mode

    # ASR Metrics
    wer_scores: List[float] = []
    cer_scores: List[float] = []
    if mode in ["asr_text_retrieval", "asr_only"]:
        results["asr"] = asr_pipeline.model.name()

        for gt_text, hyp_text in zip(all_ground_truth, _asr_hypotheses_for_wer):
            wer_scores.append(word_error_rate(gt_text, hyp_text))
            cer_scores.append(character_error_rate(gt_text, hyp_text))

        results["WER"] = sum(wer_scores) / len(wer_scores) if wer_scores else 0.0
        results["CER"] = sum(cer_scores) / len(cer_scores) if cer_scores else 0.0
        logger.info(f"ASR Metrics - WER: {results['WER']:.4f}, CER: {results['CER']:.4f}")

        if term_weights:
            tw_scores = [
                term_weighted_wer(gt, hyp, term_weights)
                for gt, hyp in zip(all_ground_truth, _asr_hypotheses_for_wer)
            ]
            results["TW_WER"] = sum(tw_scores) / len(tw_scores) if tw_scores else 0.0
            logger.info(f"Term-Weighted WER: {results['TW_WER']:.4f}")

    # Embedding model info
    if mode == "asr_text_retrieval":
        results["embedder"] = text_embedding_pipeline.model.name()
    elif mode == "audio_text_retrieval":
        results["audio_embedder"] = audio_embedding_pipeline.model.name()
        results["text_embedder"] = text_embedding_pipeline.model.name()
        # Audio↔text embedding alignment score (cosine similarity per pair)
        try:
            _alignment = embedding_alignment(_audio_emb_for_alignment, _text_emb_for_alignment)
            if _alignment is not None:
                results["embedding_alignment"] = _alignment
                logger.info(
                    "Embedding alignment — cosine mean=%.4f std=%.4f",
                    _alignment["audio_text_cosine_mean"],
                    _alignment["audio_text_cosine_std"],
                )
        except Exception as _e:
            logger.warning("Embedding alignment computation failed: %s", _e)

    # IR Metrics
    if mode != "asr_only":
        if all_relevance:
            all_relevant = all_relevance
        else:
            all_relevant = [{str(gt): 1} for gt in all_ground_truth]

        ir_metrics = compute_ir_metrics(all_retrieved, all_relevant, k_values=[1, 5, 10])
        results.update(ir_metrics)
        log_ir_metrics(results, logger, k_values=[1, 5, 10])

        # Per-query recall@5 (used in traces and WER/recall correlation)
        _per_query_recall5 = [
            recall_at_k(ret, rel, 5)
            for ret, rel in zip(all_retrieved, all_relevant)
        ]

        # WER / Recall@5 correlation (only when ASR mode and enough samples)
        _corr = wer_recall_correlation(wer_scores, _per_query_recall5)
        if _corr is not None:
            results["wer_recall5_correlation"] = _corr

        # First-relevant rank distribution
        _rank_dist, _failure_rate = first_relevant_rank_distribution(all_retrieved, all_relevant)
        results["first_relevant_rank_distribution"] = _rank_dist
        results["retrieval_failure_rate"] = _failure_rate

        # Retrieval failure mode decomposition (when traces enabled)
        if trace_limit > 0:
            _query_texts_for_err = all_hypotheses if mode == "asr_text_retrieval" else all_ground_truth
            _err_details = [
                {"query": _query_texts_for_err[_i], "retrieved": all_retrieved[_i], "relevant": all_relevant[_i]}
                for _i in range(len(all_retrieved))
            ]
            _base_failure_analysis = analyze_retrieval_failures({"details": _err_details}, top_k=k)
            # Augment with WER-based categorisation when ASR scores available
            if wer_scores and len(wer_scores) == len(_per_query_recall5):
                # Corpus doc-id set lets us split a true corpus gap (relevant doc
                # absent from the corpus) from ASR/embedding failures.
                _corpus_doc_ids = None
                if hasattr(dataset, "get_corpus"):
                    try:
                        _corpus_doc_ids = {
                            str(doc.get("doc_id", "")) for doc in dataset.get_corpus()
                        }
                        _corpus_doc_ids.discard("")
                    except Exception:
                        _corpus_doc_ids = None
                _base_failure_analysis["failure_categories"] = categorize_failures(
                    wer_scores, _per_query_recall5, _rank_dist,
                    all_relevant=all_relevant, corpus_doc_ids=_corpus_doc_ids,
                )
            results["retrieval_failure_analysis"] = _base_failure_analysis

        # Bootstrap confidence intervals (95%, 1000 iterations)
        if compute_confidence_intervals and len(all_retrieved) >= MIN_SAMPLES_FOR_CI:
            _rr_scores = [reciprocal_rank(ret, rel) for ret, rel in zip(all_retrieved, all_relevant)]
            _ndcg5_scores = [ndcg_at_k(ret, rel, 5) for ret, rel in zip(all_retrieved, all_relevant)]
            _ci_metrics = {
                "MRR": _rr_scores,
                "Recall@5": _per_query_recall5,
                "NDCG@5": _ndcg5_scores,
            }
            for _metric_name, _scores in _ci_metrics.items():
                try:
                    _lo, _hi = bootstrap_confidence_interval(
                        _scores, alpha=BOOTSTRAP_ALPHA, n_bootstrap=BOOTSTRAP_ITERATIONS
                    )
                    results[f"{_metric_name}_ci"] = (_lo, _hi)
                except Exception as _e:
                    logger.warning("CI computation failed for %s: %s", _metric_name, _e)
            logger.info("Bootstrap CI (95%%) computed for %d samples", len(all_retrieved))

        _stage_times["metrics_s"] = time.perf_counter() - _t_phase
        # Hand the IR-stage intermediates to the answer-gen / finalize stages.
        s.metrics_all_relevant = all_relevant
        s.per_query_recall5 = _per_query_recall5

    s.wer_scores = wer_scores
    s.cer_scores = cer_scores
    s.results = results


@register_stage_handler("answer_gen", self_timed=True)
def _stage_answer_gen(s: _PhasedState) -> None:
    """PHASE 5: RAG answer generation + per-query traces (retrieval modes only).

    Reads the IR intermediates the metrics stage stored on the state; writes
    ``answer_generation`` and ``query_traces`` into ``s.results``.
    """
    if s.mode == "asr_only":
        return
    _t = time.perf_counter()
    s.ans_detail_by_qid = _generate_answer_details(s, s.results, s.metrics_all_relevant)
    _build_query_traces(
        s, s.results, s.metrics_all_relevant, s.wer_scores, s.cer_scores,
        s.per_query_recall5, s.ans_detail_by_qid,
    )
    s.stage_times["answer_gen_s"] = (
        s.stage_times.get("answer_gen_s", 0.0) + (time.perf_counter() - _t)
    )


@register_stage_handler("finalize", self_timed=True)
def _stage_finalize(s: _PhasedState) -> None:
    """PHASE 6: LLM judge (retrieval modes) + latency summary. Terminal node."""
    if s.mode != "asr_only":
        _t = time.perf_counter()
        _run_judge(s, s.results, s.metrics_all_relevant, s.per_query_recall5)
        s.stage_times["judge_s"] = (
            s.stage_times.get("judge_s", 0.0) + (time.perf_counter() - _t)
        )
    s.stage_times["total_s"] = time.perf_counter() - s.t_total
    s.results["latency"] = s.stage_times
    logger.info(
        "Stage latency — asr=%.1fs embed=%.1fs retrieve=%.1fs total=%.1fs",
        s.stage_times.get("asr_s", 0),
        s.stage_times.get("embedding_s", 0),
        s.stage_times.get("retrieval_s", 0),
        s.stage_times["total_s"],
    )


# Stage handlers register themselves via @register_stage_handler (see stage_registry);
# the executor discovers them by name. Timing policy (self_timed / time_key) lives on
# each StageSpec, so adding a stage no longer means editing a central dispatch dict.


# Stage id -> the _PhasedState attribute holding the pipeline whose model that stage runs.
_STAGE_PIPELINE_ATTR = {
    "asr": "asr_pipeline",
    "text_embedding": "text_embedding_pipeline",
    "audio_embedding": "audio_embedding_pipeline",
}


def _stage_model(state: "_PhasedState", stage: str):
    """The model a stage runs, or None for model-free stages (fusion/metrics).

    The retrieval stage's model is its cross-encoder reranker (when configured); it is used
    only there, so it can be freed once retrieval completes.
    """
    if stage == "retrieval":
        rp = getattr(state, "retrieval_pipeline", None)
        return getattr(rp, "reranker", None) if rp is not None else None
    attr = _STAGE_PIPELINE_ATTR.get(stage)
    pipe = getattr(state, attr, None) if attr else None
    return getattr(pipe, "model", None) if pipe is not None else None


def _plan_stage_offloads(state: "_PhasedState", flat_nodes, query_opt_enabled: bool):
    """Map each stage position to the models whose LAST use is there (so they can be freed).

    Keyed by model *instance* so a model shared across stages (e.g. a multimodal embedder
    used for both audio and text) is released only after its final stage. The text
    embedder is excluded when query optimization is on, since query-opt may re-embed.
    """
    if not state.offload_after_stage or state.service_provider is None:
        return {}
    excluded = set()
    if query_opt_enabled:
        te = _stage_model(state, "text_embedding")
        if te is not None:
            excluded.add(id(te))
    last: Dict[int, tuple] = {}
    for pos, node in enumerate(flat_nodes):
        model = _stage_model(state, node.stage)
        if model is not None and id(model) not in excluded:
            last[id(model)] = (pos, model)
    by_pos: Dict[int, list] = {}
    for pos, model in last.values():
        by_pos.setdefault(pos, []).append(model)
    return by_pos


def _execute_stage_graph(
    state: "_PhasedState", stage_graph, query_opt_config,
    text_embedding_pipeline, retrieval_pipeline,
) -> None:
    """Dispatch each DAG node to its handler in topological order.

    The stage graph is the single source of truth for what runs and in what
    order. The terminal "metrics" node tracks its own per-sub-phase timing;
    query optimization is a documented transform inserted after the source level
    (no graph node, since it can bypass embedding+retrieval). When a provider +
    on_finish policy apply, each stage's model is released after its last use.
    """
    query_opt_enabled = query_opt_config is not None and getattr(
        query_opt_config, "enabled", False
    )
    flat_nodes = [node for level in stage_graph.topological_levels() for node in level]
    offloads = _plan_stage_offloads(state, flat_nodes, query_opt_enabled)
    pos = 0
    for level_idx, level in enumerate(stage_graph.topological_levels()):
        for node in level:
            spec = get_stage_spec(node.stage)
            if spec.self_timed:
                spec.fn(state)
            else:
                _t = time.perf_counter()
                spec.fn(state)
                state.stage_times[spec.time_key] += time.perf_counter() - _t
            for model in offloads.get(pos, ()):
                try:
                    state.service_provider.release_model_instance(model)
                    logger.info("offloaded model after stage '%s'", node.stage)
                except Exception as exc:  # never let offload break the run
                    logger.warning("offload after stage '%s' failed: %s", node.stage, exc)
            pos += 1

        # Query optimization runs once, right after the query texts are produced.
        if level_idx == 0 and query_opt_enabled:
            _run_query_optimization_stage(
                state, query_opt_config, text_embedding_pipeline, retrieval_pipeline
            )


def _run_query_optimization_stage(
    state: "_PhasedState", query_opt_config, text_embedding_pipeline, retrieval_pipeline,
) -> None:
    """PHASE 1.5: optimize query texts (may bypass embedding+retrieval)."""
    _log_phase(f"PHASE 1.5: Query Optimization ({query_opt_config.method})")
    state.cb(
        "phase_1_5_query_opt", 0, state.total,
        f"Phase 1.5: Query optimization ({query_opt_config.method})",
    )
    _t = time.perf_counter()
    (
        state.all_hypotheses,
        state.all_ground_truth,
        state.all_results_with_scores,
        _qopt_retrieved,
        state.query_opt_bypassed,
    ) = _run_query_optimization_phase(
        state.mode,
        state.all_hypotheses,
        state.all_ground_truth,
        state.all_results_with_scores,
        query_opt_config,
        text_embedding_pipeline,
        retrieval_pipeline,
        state.k,
    )
    if state.query_opt_bypassed:
        state.all_retrieved = _qopt_retrieved
    state.stage_times["query_opt_s"] += time.perf_counter() - _t


def evaluate_phased(
    dataset: QueryDataset,
    retrieval_pipeline: Optional[RetrievalPipelineProtocol] = None,
    asr_pipeline: Optional[ASRPipelineProtocol] = None,
    text_embedding_pipeline: Optional[TextEmbeddingPipelineProtocol] = None,
    audio_embedding_pipeline: Optional[AudioEmbeddingPipelineProtocol] = None,
    cache_manager: Optional[CacheManager] = None,
    k: int = 10,
    batch_size: int = 32,
    trace_limit: int = 0,
    features: Optional["PhasedFeatures"] = None,
    num_workers: int = 0,
    checkpoint_interval: int = 500,
    experiment_id: Optional[str] = None,
    resume_from_checkpoint: bool = True,
    progress_callback: Optional[Callable[[str, int, int, str], None]] = None,
    oracle_mode: bool = False,
    service_provider: Any = None,
    offload_policy: str = "never",
    context: Optional["EvaluationContext"] = None,
) -> PhasedResults:
    """Optimized phased evaluation - processes entire dataset in phases.
    
    Runs evaluation in four sequential phases for efficiency:
    
    - Phase 1: ASR - Transcribe all audio samples (or audio embedding)
    - Phase 2: Embedding - Embed all transcriptions  
    - Phase 3: Retrieval - Perform all vector searches
    - Phase 4: Metrics - Compute all evaluation metrics
    
    Args:
        dataset: QueryDataset instance containing audio samples and ground truth.
        retrieval_pipeline: RetrievalPipeline instance for vector search.
        asr_pipeline: Optional ASRPipeline instance for transcription.
        text_embedding_pipeline: Optional TextEmbeddingPipeline instance.
        audio_embedding_pipeline: Optional AudioEmbeddingPipeline instance.
        cache_manager: Optional CacheManager for checkpointing.
        k: Number of retrieval results to return. Default: 10.
        batch_size: Batch size for processing. Default: 32.
        trace_limit: Limit number of samples (0 = no limit). Default: 0.
        features: Optional PhasedFeatures bundling the default-off knobs (LLM
            judge, answer generation, query optimization, embedding fusion, domain
            term weights, bootstrap CIs). None = all features disabled.
        num_workers: Number of DataLoader workers. Default: 0.
        checkpoint_interval: Save checkpoint every N samples. Default: 500.
        experiment_id: Unique ID for this experiment (for checkpointing).
        resume_from_checkpoint: Whether to resume from existing checkpoint.
        context: Optional EvaluationContext bundling all of the pipeline + execution
            params above. When provided it supersedes the individual keyword args —
            the low-parameter entry point.

    Returns:
        Dictionary of evaluation metrics including:
        - pipeline_mode: Mode used (asr_text_retrieval, audio_emb_retrieval, etc.)
        - WER, CER: Word/Character error rates (ASR modes)
        - MRR: Mean Reciprocal Rank
        - MAP: Mean Average Precision
        - Recall@k: Recall at k for various k values
        - NDCG@k: Normalized Discounted Cumulative Gain at k
        - total_samples: Number of samples processed
        - duration_seconds: Total evaluation time
    
    Raises:
        ValueError: If neither audio_embedding_pipeline nor asr_pipeline provided.
    
    Examples:
        Basic usage with ASR and text retrieval::
        
            >>> from evaluator.evaluation import evaluate_phased
            >>> from evaluator.pipeline import create_pipeline_from_config
            >>> 
            >>> bundle = create_pipeline_from_config(config, cache_manager)
            >>> results = evaluate_phased(
            ...     dataset=dataset,
            ...     retrieval_pipeline=bundle.retrieval_pipeline,
            ...     asr_pipeline=bundle.asr_pipeline,
            ...     text_embedding_pipeline=bundle.text_embedding_pipeline,
            ...     k=5,
            ...     batch_size=32
            ... )
        
        Interpreting results::
        
            >>> print(f"Pipeline: {results['pipeline_mode']}")
            Pipeline: asr_text_retrieval
            >>> print(f"ASR Quality - WER: {results['WER']:.2%}")
            ASR Quality - WER: 12.34%
            >>> print(f"Retrieval - MRR: {results['MRR']:.4f}")
            Retrieval - MRR: 0.7523
            >>> print(f"Recall@5: {results['Recall@5']:.4f}")
            Recall@5: 0.8912
        
        With checkpointing for long evaluations::
        
            >>> results = evaluate_phased(
            ...     dataset=large_dataset,
            ...     retrieval_pipeline=retrieval,
            ...     asr_pipeline=asr,
            ...     text_embedding_pipeline=text_emb,
            ...     cache_manager=cache,
            ...     checkpoint_interval=100,
            ...     experiment_id="long_eval_run",
            ...     resume_from_checkpoint=True
            ... )
    """
    # An EvaluationContext bundles the pipeline + execution params; when supplied
    # it supersedes the individual keyword arguments (which keep their defaults for
    # direct callers). This is the low-parameter entry point.
    if context is not None:
        retrieval_pipeline = context.retrieval_pipeline
        asr_pipeline = context.asr_pipeline
        text_embedding_pipeline = context.text_embedding_pipeline
        audio_embedding_pipeline = context.audio_embedding_pipeline
        cache_manager = context.cache_manager
        k = context.k
        batch_size = context.batch_size
        trace_limit = context.trace_limit
        num_workers = context.num_workers
        checkpoint_interval = context.checkpoint_interval
        experiment_id = context.experiment_id
        resume_from_checkpoint = context.resume_from_checkpoint
        progress_callback = context.progress_callback
        oracle_mode = context.oracle_mode
        features = context.features

    # Optional feature configs are bundled in `features` (all default-off).
    features = features or PhasedFeatures()
    judge_config = features.judge_config
    answer_gen_config = features.answer_gen_config
    query_opt_config = features.query_opt_config
    embedding_fusion_config = features.embedding_fusion_config
    term_weights = features.term_weights
    compute_confidence_intervals = features.compute_confidence_intervals

    # Determine mode
    mode = detect_pipeline_mode(
        retrieval_pipeline,
        asr_pipeline,
        text_embedding_pipeline,
        audio_embedding_pipeline,
    )
    logger.info("Evaluation mode: %s (Phased)", PIPELINE_MODE_LABELS[mode])

    stage_graph = build_stage_graph(
        mode,
        embedding_fusion_enabled=bool(
            embedding_fusion_config is not None and embedding_fusion_config.enabled
        ),
    )
    stage_levels = [[node.id for node in level] for level in stage_graph.topological_levels()]
    logger.info("Execution DAG mode=%s levels=%s", mode, stage_levels)
    
    logger.info(f"Dataset size: {len(dataset)}, Batch size: {batch_size}, k: {k}")

    _total = len(dataset)
    _cb = progress_callback or (lambda *_: None)
    _cb("init", 0, _total, f"Starting {mode} evaluation ({_total} samples)")

    _t_total = time.perf_counter()

    # Execution context shared across stage handlers (see _PhasedState).
    state = _PhasedState(
        dataset=dataset,
        mode=mode,
        retrieval_pipeline=retrieval_pipeline,
        asr_pipeline=asr_pipeline,
        text_embedding_pipeline=text_embedding_pipeline,
        audio_embedding_pipeline=audio_embedding_pipeline,
        cache_manager=cache_manager,
        k=k,
        batch_size=batch_size,
        num_workers=num_workers,
        checkpoint_interval=checkpoint_interval,
        experiment_id=experiment_id,
        resume_from_checkpoint=resume_from_checkpoint,
        oracle_mode=oracle_mode,
        embedding_fusion_config=embedding_fusion_config,
        query_opt_config=query_opt_config,
        answer_gen_config=answer_gen_config,
        judge_config=judge_config,
        trace_limit=trace_limit,
        term_weights=term_weights,
        compute_confidence_intervals=compute_confidence_intervals,
        total=_total,
        cb=_cb,
        t_total=_t_total,
        service_provider=service_provider,
        offload_after_stage=(service_provider is not None and offload_policy == "on_finish"),
    )
    state.stage_times = {"asr_s": 0.0, "query_opt_s": 0.0, "embedding_s": 0.0, "retrieval_s": 0.0}

    # DAG-driven execution: the stage graph drives what runs and in what order.
    _execute_stage_graph(
        state, stage_graph, query_opt_config, text_embedding_pipeline, retrieval_pipeline
    )

    results = state.results

    if cache_manager and experiment_id:
        try:
            import os
            checkpoint_path = cache_manager._get_cache_path("checkpoints", f"{experiment_id}_phased", ".json")
            if checkpoint_path.exists():
                os.remove(checkpoint_path)
        except OSError as exc:
            logger.warning("Failed to clean up checkpoint file: %s", exc)

    if cache_manager:
        log_cache_stats(cache_manager, logger)

    _cb("done", _total, _total, "Evaluation complete")
    return results


def evaluate_from_bundle(
    dataset: QueryDataset,
    bundle: "PipelineBundle",
    config: "EvaluationConfig",
    *,
    cache_manager: Optional[CacheManager] = None,
    progress_callback: Optional[Callable[[str, int, int, str], None]] = None,
) -> PhasedResults:
    """Convenience wrapper: extract params from bundle + config.

    Reduces a 16-param call to 4 args.  Fusion, judge, answer generation, and
    query optimization are enabled only when their respective ``config.*. enabled``
    flag is True; call ``evaluate_phased`` directly to force any of them on
    regardless of config flags.
    """
    _term_weights: Optional[Dict[str, float]] = None
    _tww_path = getattr(config, "domain_term_weights_file", None)
    if _tww_path:
        try:
            _term_weights = load_term_weights(domain="", path=_tww_path)
            logger.info("Loaded %d domain term weights from %s", len(_term_weights), _tww_path)
        except FileNotFoundError as _e:
            logger.warning("domain_term_weights_file not found: %s", _e)

    features = PhasedFeatures(
        judge_config=config.judge if config.judge.enabled else None,
        answer_gen_config=config.answer_generation if config.answer_generation.enabled else None,
        query_opt_config=config.query_optimization if config.query_optimization.enabled else None,
        embedding_fusion_config=config.embedding_fusion if config.embedding_fusion.enabled else None,
        term_weights=_term_weights,
        compute_confidence_intervals=getattr(config, "compute_confidence_intervals", False),
    )
    # Per-stage model offload: free each stage's model after its last use. Guarded so the
    # oracle baseline re-run below (which reuses these pipelines) still has its models — the
    # first call must not offload when an oracle pass follows.
    _offload_policy = getattr(getattr(config, "service_runtime", None), "offload_policy", "never")
    _oracle_will_run = (
        getattr(config, "compute_oracle_baseline", False) and bundle.mode == "asr_text_retrieval"
    )
    _shared_kwargs = dict(
        retrieval_pipeline=bundle.retrieval_pipeline,
        asr_pipeline=bundle.asr_pipeline,
        text_embedding_pipeline=bundle.text_embedding_pipeline,
        audio_embedding_pipeline=bundle.audio_embedding_pipeline,
        cache_manager=cache_manager,
        k=config.vector_db.k,
        batch_size=config.data.batch_size,
        trace_limit=config.data.trace_limit,
        num_workers=config.data.num_workers,
        checkpoint_interval=config.checkpoint_interval if config.checkpoint_enabled else 0,
        experiment_id=config.experiment_name,
        resume_from_checkpoint=getattr(config, 'resume_from_checkpoint', True),
        progress_callback=progress_callback,
        service_provider=bundle.service_provider,
    )
    results = evaluate_phased(
        dataset=dataset, features=features,
        offload_policy="never" if _oracle_will_run else _offload_policy,
        **_shared_kwargs,
    )

    if getattr(config, "compute_oracle_baseline", False) and results.get("pipeline_mode") == "asr_text_retrieval":
        # The oracle baseline only contributes MRR / Recall@5 / NDCG@5 to the
        # degradation factor. Skip the extras that don't affect those but cost
        # real time/LLM calls: judge, answer generation, traces, bootstrap CI.
        from dataclasses import replace as _replace
        oracle_features = _replace(features, judge_config=None, answer_gen_config=None, compute_confidence_intervals=False)
        oracle_results = evaluate_phased(
            dataset=dataset, oracle_mode=True, features=oracle_features,
            offload_policy=_offload_policy,  # last use of these pipelines → safe to offload
            **dict(_shared_kwargs, trace_limit=0),
        )
        actual_mrr = results.get("MRR", 0.0)
        oracle_mrr = oracle_results.get("MRR", 0.0)
        results["oracle_MRR"] = oracle_mrr
        results["oracle_Recall@5"] = oracle_results.get("Recall@5", 0.0)
        results["oracle_NDCG@5"] = oracle_results.get("NDCG@5", 0.0)
        results["asr_degradation_factor"] = (actual_mrr / oracle_mrr) if oracle_mrr > 0 else None

    return results


__all__ = ["evaluate_phased", "evaluate_from_bundle", "PhasedFeatures", "EvaluationContext"]
