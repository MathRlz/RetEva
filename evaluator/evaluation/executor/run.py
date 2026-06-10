"""Run entry points: ``run_graph`` + ``run_from_bundle``.

Moved out of the former ``evaluation/phased.py`` (Phase 1, X9). ``run_graph`` is
the core DAG run loop (build graph → seed run state → execute stage graph → collect
results); ``run_from_bundle`` is the config-driven convenience wrapper.
"""

from __future__ import annotations

import time
from typing import Callable, Optional, Dict, Any

from ...datasets import QueryDataset
from ...pipeline import (
    ASRPipelineProtocol,
    TextEmbeddingPipelineProtocol,
    AudioEmbeddingPipelineProtocol,
    RetrievalPipelineProtocol,
)
from ...storage.cache import CacheManager
from ...logging_config import get_logger, log_cache_stats
from ...metrics.domain_terms import load_term_weights
from ...pipeline.run_graph import _build_run_graph
from ..helpers import detect_pipeline_mode, PIPELINE_MODE_LABELS
from ..result_schema import RunResults
from .state import RunState, RunFeatures, EvaluationContext
from .engine import _execute_stage_graph

logger = get_logger(__name__)


def run_graph(
    dataset: QueryDataset,
    retrieval_pipeline: Optional[RetrievalPipelineProtocol] = None,
    asr_pipeline: Optional[ASRPipelineProtocol] = None,
    text_embedding_pipeline: Optional[TextEmbeddingPipelineProtocol] = None,
    audio_embedding_pipeline: Optional[AudioEmbeddingPipelineProtocol] = None,
    cache_manager: Optional[CacheManager] = None,
    k: int = 10,
    batch_size: int = 32,
    trace_limit: int = 0,
    features: Optional["RunFeatures"] = None,
    num_workers: int = 0,
    checkpoint_interval: int = 500,
    experiment_id: Optional[str] = None,
    resume_from_checkpoint: bool = True,
    progress_callback: Optional[Callable[[str, int, int, str], None]] = None,
    oracle_mode: bool = False,
    service_provider: Any = None,
    offload_policy: str = "never",
    eval_config: Any = None,
    load_info: Any = None,
    graph_override: Any = None,
    context: Optional["EvaluationContext"] = None,
) -> RunResults:
    """Optimized phased evaluation - processes entire dataset in phases.

    Runs evaluation by dispatching the stage-graph DAG node by node (the graph,
    not fixed phases, is the source of truth). A typical retrieval graph wires:

    - ASR node — transcribe all audio samples (or audio-embedding node)
    - embedding node — embed all transcriptions
    - retrieval node — perform all vector searches
    - metrics node — compute all evaluation metrics

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
        features: Optional RunFeatures bundling the default-off knobs (LLM
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

            >>> from evaluator.evaluation import run_graph
            >>> from evaluator.pipeline import create_pipeline_from_config
            >>>
            >>> bundle = create_pipeline_from_config(config, cache_manager)
            >>> results = run_graph(
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

            >>> results = run_graph(
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
    features = features or RunFeatures()
    judge_config = features.judge_config
    answer_gen_config = features.answer_gen_config
    query_opt_config = features.query_opt_config
    query_correction_config = features.query_correction_config
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
    logger.info("Evaluation mode: %s (DAG)", PIPELINE_MODE_LABELS[mode])

    stage_graph = _build_run_graph(
        mode,
        graph_override=graph_override,
        embedding_fusion_config=embedding_fusion_config,
        query_opt_config=query_opt_config,
        query_correction_config=query_correction_config,
        retrieval_pipeline=retrieval_pipeline,
        eval_config=eval_config,
    )
    stage_levels = [
        [node.id for node in level] for level in stage_graph.topological_levels()
    ]
    logger.info("Execution DAG mode=%s levels=%s", mode, stage_levels)

    logger.info(f"Dataset size: {len(dataset)}, Batch size: {batch_size}, k: {k}")

    _total = len(dataset)
    _cb = progress_callback or (lambda *_: None)
    _cb("init", 0, _total, f"Starting {mode} evaluation ({_total} samples)")

    _t_total = time.perf_counter()

    # Multi-dataset runtime (B1): when the config carries a `datasets:` map, load each source so
    # per-node dataset_source handlers can pick theirs. Empty otherwise (single-source unchanged).
    dataset_sources: Dict[str, Any] = {}
    disable_ir_metrics, join_warning = False, ""
    if eval_config is not None and getattr(
        getattr(eval_config, "data", None), "datasets", None
    ):
        from ...datasets import load_runtime_datasets, validate_dataset_join

        dataset_sources = load_runtime_datasets(eval_config)
        # B5: validate the cross-source join (questions↔corpus doc_id overlap); disjoint → IR off.
        cfg_sources = getattr(eval_config.data, "datasets", {}) or {}
        q_id = next(
            (
                s
                for s, e in cfg_sources.items()
                if (e or {}).get("role") in ("questions", "both")
            ),
            None,
        )
        c_id = next(
            (
                s
                for s, e in cfg_sources.items()
                if (e or {}).get("role") in ("corpus", "both")
            ),
            None,
        )
        if q_id and c_id and q_id != c_id and {q_id, c_id} <= set(dataset_sources):
            join = validate_dataset_join(dataset_sources[q_id], dataset_sources[c_id])
            if join["disjoint"]:
                disable_ir_metrics, join_warning = True, join["warning"]
                logger.warning("dataset join: %s", join_warning)

    # Execution context shared across stage handlers (see RunState).
    state = RunState(
        dataset=dataset,
        mode=mode,
        dataset_sources=dataset_sources,
        disable_ir_metrics=disable_ir_metrics,
        join_warning=join_warning,
        retrieval_pipeline=retrieval_pipeline,
        asr_pipeline=asr_pipeline,
        text_embedding_pipeline=text_embedding_pipeline,
        audio_embedding_pipeline=audio_embedding_pipeline,
        cache_manager=cache_manager,
        config=eval_config,
        load_info=load_info,
        k=k,
        batch_size=batch_size,
        num_workers=num_workers,
        checkpoint_interval=checkpoint_interval,
        experiment_id=experiment_id,
        resume_from_checkpoint=resume_from_checkpoint,
        oracle_mode=oracle_mode,
        embedding_fusion_config=embedding_fusion_config,
        query_opt_config=query_opt_config,
        query_correction_config=query_correction_config,
        answer_gen_config=answer_gen_config,
        judge_config=judge_config,
        trace_limit=trace_limit,
        term_weights=term_weights,
        compute_confidence_intervals=compute_confidence_intervals,
        total=_total,
        cb=_cb,
        t_total=_t_total,
        service_provider=service_provider,
        offload_after_stage=(
            service_provider is not None and offload_policy == "on_finish"
        ),
        rerank_in_graph=any(n.stage == "rerank" for n in stage_graph.nodes),
    )
    state.stage_times = {
        "asr_s": 0.0,
        "query_opt_s": 0.0,
        "correction_s": 0.0,
        "embedding_s": 0.0,
        "retrieval_s": 0.0,
    }

    # DAG-driven execution: the stage graph drives what runs and in what order.
    _execute_stage_graph(state, stage_graph, query_opt_config)

    results = state.results

    if cache_manager and experiment_id:
        try:
            import os

            checkpoint_path = cache_manager._get_cache_path(
                "checkpoints", f"{experiment_id}_phased", ".json"
            )
            if checkpoint_path.exists():
                os.remove(checkpoint_path)
        except OSError as exc:
            logger.warning("Failed to clean up checkpoint file: %s", exc)

    if cache_manager:
        log_cache_stats(cache_manager, logger)

    _cb("done", _total, _total, "Evaluation complete")
    return results


def run_from_bundle(
    dataset: QueryDataset,
    bundle: "PipelineBundle",
    config: "EvaluationConfig",
    *,
    cache_manager: Optional[CacheManager] = None,
    progress_callback: Optional[Callable[[str, int, int, str], None]] = None,
    load_info: Optional[Dict[str, Any]] = None,
) -> RunResults:
    """Convenience wrapper: extract params from bundle + config.

    Reduces a 16-param call to 4 args.  Fusion, judge, answer generation, and
    query optimization are enabled only when their respective ``config.*. enabled``
    flag is True; call ``run_graph`` directly to force any of them on
    regardless of config flags.
    """
    _term_weights: Optional[Dict[str, float]] = None
    _tww_path = getattr(config, "domain_term_weights_file", None)
    if _tww_path:
        try:
            _term_weights = load_term_weights(domain="", path=_tww_path)
            logger.info(
                "Loaded %d domain term weights from %s", len(_term_weights), _tww_path
            )
        except FileNotFoundError as _e:
            logger.warning("domain_term_weights_file not found: %s", _e)

    features = RunFeatures(
        judge_config=config.judge if config.judge.enabled else None,
        answer_gen_config=(
            config.answer_generation if config.answer_generation.enabled else None
        ),
        query_opt_config=(
            config.query_optimization if config.query_optimization.enabled else None
        ),
        query_correction_config=(
            config.query_correction
            if getattr(config, "query_correction", None)
            and config.query_correction.enabled
            else None
        ),
        embedding_fusion_config=(
            config.embedding_fusion if config.embedding_fusion.enabled else None
        ),
        term_weights=_term_weights,
        compute_confidence_intervals=getattr(
            config, "compute_confidence_intervals", False
        ),
    )
    # Per-stage model offload: free each stage's model after its last use. Guarded so the
    # oracle baseline re-run below (which reuses these pipelines) still has its models — the
    # first call must not offload when an oracle pass follows.
    _offload_policy = getattr(
        getattr(config, "service_runtime", None), "offload_policy", "never"
    )
    _oracle_will_run = (
        getattr(config, "compute_oracle_baseline", False)
        and bundle.mode == "asr_text_retrieval"
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
        checkpoint_interval=(
            config.checkpoint_interval if config.checkpoint_enabled else 0
        ),
        experiment_id=config.experiment_name,
        resume_from_checkpoint=getattr(config, "resume_from_checkpoint", True),
        progress_callback=progress_callback,
        service_provider=bundle.service_provider,
        eval_config=config,
        load_info=load_info,
        graph_override=getattr(config, "graph_override", None),
    )
    results = run_graph(
        dataset=dataset,
        features=features,
        offload_policy="never" if _oracle_will_run else _offload_policy,
        **_shared_kwargs,
    )

    if (
        getattr(config, "compute_oracle_baseline", False)
        and results.get("pipeline_mode") == "asr_text_retrieval"
    ):
        # The oracle baseline only contributes MRR / Recall@5 / NDCG@5 to the
        # degradation factor. Skip the extras that don't affect those but cost
        # real time/LLM calls: judge, answer generation, traces, bootstrap CI.
        from dataclasses import replace as _replace

        oracle_features = _replace(
            features,
            judge_config=None,
            answer_gen_config=None,
            compute_confidence_intervals=False,
        )
        oracle_results = run_graph(
            dataset=dataset,
            oracle_mode=True,
            features=oracle_features,
            offload_policy=_offload_policy,  # last use of these pipelines → safe to offload
            **dict(_shared_kwargs, trace_limit=0),
        )
        actual_mrr = results.get("MRR", 0.0)
        oracle_mrr = oracle_results.get("MRR", 0.0)
        results["oracle_MRR"] = oracle_mrr
        results["oracle_Recall@5"] = oracle_results.get("Recall@5", 0.0)
        results["oracle_NDCG@5"] = oracle_results.get("NDCG@5", 0.0)
        results["asr_degradation_factor"] = (
            (actual_mrr / oracle_mrr) if oracle_mrr > 0 else None
        )

    return results
