"""Run entry points: ``run_graph`` + ``run_from_bundle``.

Moved out of the former ``evaluation/phased.py`` (Phase 1, X9). ``run_graph`` is
the core DAG run loop (build graph → seed run state → execute stage graph → collect
results); ``run_from_bundle`` is the config-driven convenience wrapper.
"""

from __future__ import annotations

import time
from typing import Callable, Optional, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ...config import EvaluationConfig
    from ...pipeline.types import PipelineBundle

from ...datasets import QueryDataset
from ...storage.cache import CacheManager
from ...logging_config import get_logger, log_cache_stats, node_logger
from ...metrics.domain_terms import load_term_weights
from ...pipeline.graph.modes import build_run_graph
from ..helpers import detect_graph_template
from ...pipeline.graph.templates import GRAPH_TEMPLATES
from ..stage_registry import validate_graph_handlers
from ..handlers.rag import drop_mirrored_top_level_keys
from ..result_schema import RunResults
from .state import RunState, RunFeatures, EvaluationContext
from .engine import _execute_stage_graph

logger = get_logger(__name__)


def _node_kind(node) -> str:
    """The legacy node-kind name for graph-flag derivation (operator-abstraction): an
    operator node resolves to its old stage name, a pre-collapse node is itself."""
    from ...pipeline.graph.operators import node_kind

    return node_kind(node.stage, node.params)


def run_graph(
    dataset: QueryDataset,
    context: "EvaluationContext",
    *,
    service_provider: Any = None,
    offload_policy: str = "never",
    eval_config: Any = None,
    load_info: Any = None,
    graph_override: Any = None,
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
        context: EvaluationContext bundling the pipelines + execution params (D1/F5 — the
            sole entry contract: retrieval/asr/text-emb/audio-emb pipelines, cache_manager,
            k, batch_size, trace_limit, num_workers, checkpoint_interval, experiment_id,
            resume_from_checkpoint, progress_callback, oracle_mode, and ``features``).
        service_provider: Optional model service provider (offload coordination).
        offload_policy: "never" | "on_finish" — free each stage's model after last use.
        eval_config: The EvaluationConfig (mode detection, multi-dataset sources, provenance).
        load_info: Optional model load metadata for the report.
        graph_override: Optional explicit stage-graph spec replacing the default for the mode.

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
            >>> from evaluator.evaluation.executor.state import EvaluationContext
            >>> from evaluator.pipeline import create_pipeline_from_config
            >>>
            >>> bundle = create_pipeline_from_config(config, cache_manager)
            >>> context = EvaluationContext(
            ...     retrieval_pipeline=bundle.retrieval_pipeline,
            ...     asr_pipeline=bundle.asr_pipeline,
            ...     text_embedding_pipeline=bundle.text_embedding_pipeline,
            ...     k=5,
            ...     batch_size=32,
            ... )
            >>> results = run_graph(dataset, context, eval_config=config)

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

            >>> context = EvaluationContext(
            ...     retrieval_pipeline=retrieval,
            ...     asr_pipeline=asr,
            ...     text_embedding_pipeline=text_emb,
            ...     cache_manager=cache,
            ...     checkpoint_interval=100,
            ...     experiment_id="long_eval_run",
            ...     resume_from_checkpoint=True,
            ... )
            >>> results = run_graph(large_dataset, context, eval_config=config)
    """
    # The EvaluationContext is the sole source of the pipeline + execution params (D1/F5).
    # Only the fields used directly in this function are unpacked; the rest are read from
    # ``context`` in _setup_execution_context (F6).
    retrieval_pipeline = context.retrieval_pipeline
    asr_pipeline = context.asr_pipeline
    text_embedding_pipeline = context.text_embedding_pipeline
    audio_embedding_pipeline = context.audio_embedding_pipeline
    cache_manager = context.cache_manager
    k = context.k
    batch_size = context.batch_size
    trace_limit = context.trace_limit
    experiment_id = context.experiment_id
    progress_callback = context.progress_callback
    features = context.features or RunFeatures()

    # Determine mode + build the stage graph (the source of truth for what runs). The config's
    # graph template wins — cross-modal audio_emb builds a text pipeline for the corpus, which
    # pipeline-presence detection would misread as audio_text fusion.
    configured_mode = None
    if eval_config is not None and getattr(eval_config, "model", None) is not None:
        from ...pipeline.graph.modes import _config_template

        configured_mode = _config_template(eval_config)
    mode = detect_graph_template(
        retrieval_pipeline,
        asr_pipeline,
        text_embedding_pipeline,
        audio_embedding_pipeline,
        configured_mode=configured_mode,
    )
    logger.info("Evaluation template: %s (DAG)", GRAPH_TEMPLATES.get(mode, mode))
    stage_graph = build_run_graph(
        mode,
        graph_override=graph_override,
        embedding_fusion_config=features.embedding_fusion_config,
        query_opt_config=features.query_opt_config,
        query_correction_config=features.query_correction_config,
        retrieval_pipeline=retrieval_pipeline,
        eval_config=eval_config,
        trace_limit=trace_limit,
    )
    stage_levels = [
        [node.id for node in level] for level in stage_graph.topological_levels()
    ]
    node_logger.info("Execution DAG mode=%s levels=%s", mode, stage_levels)
    logger.info(f"Dataset size: {len(dataset)}, Batch size: {batch_size}, k: {k}")
    # Pre-flight (M3): fail a typo'd/unregistered node type before any heavy work.
    validate_graph_handlers(stage_graph)

    _total = len(dataset)
    _cb = progress_callback or (lambda *_: None)
    _cb("init", 0, _total, f"Starting {mode} evaluation ({_total} samples)")

    # Seed the run state (M2: setup extracted out of the DAG flow).
    state = _setup_execution_context(
        context=context,
        features=features,
        dataset=dataset,
        mode=mode,
        stage_graph=stage_graph,
        eval_config=eval_config,
        load_info=load_info,
        total=_total,
        cb=_cb,
        t_total=time.perf_counter(),
        service_provider=service_provider,
        offload_policy=offload_policy,
    )

    # DAG-driven execution: the stage graph drives what runs and in what order. With
    # ``streaming.window_size`` set, run the query side window-by-window (3a) — same RunState,
    # bounded memory; else the whole-dataset pass.
    _window = getattr(getattr(eval_config, "streaming", None), "window_size", None)
    if _window:
        from .streaming import execute_windowed

        execute_windowed(state, stage_graph, int(_window))
    else:
        _execute_stage_graph(state, stage_graph, features.query_opt_config)

    _finalize_run(cache_manager, experiment_id)
    # G5: traces/judge are consolidated into report['traces']/['judge'] during finalize; drop
    # the duplicate top-level keys at the output boundary (sinks read them during the run).
    drop_mirrored_top_level_keys(state.results)
    _cb("done", _total, _total, "Evaluation complete")
    return state.results


def _load_multi_dataset_sources(eval_config: Any):
    """Multi-dataset runtime (B1): load each source of a `datasets:` map so per-node
    dataset_source handlers can pick theirs, and validate the cross-source join (B5 —
    disjoint questions↔corpus doc_id namespaces disable the IR metrics).

    Returns ``(dataset_sources, disable_ir_metrics, join_warning)``; the empty default
    in single-source mode (no config / no map) leaves the run unchanged."""
    dataset_sources: Dict[str, Any] = {}
    disable_ir_metrics, join_warning = False, ""
    if eval_config is None or not getattr(
        getattr(eval_config, "data", None), "datasets", None
    ):
        return dataset_sources, disable_ir_metrics, join_warning
    from ...datasets import load_runtime_datasets, validate_dataset_join

    dataset_sources = load_runtime_datasets(eval_config)
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
    return dataset_sources, disable_ir_metrics, join_warning


def _setup_execution_context(
    *,
    context: "EvaluationContext",
    features: "RunFeatures",
    dataset: Any,
    mode: str,
    stage_graph: Any,
    eval_config: Any,
    load_info: Any,
    total: int,
    cb: Callable,
    t_total: float,
    service_provider: Any,
    offload_policy: str,
) -> RunState:
    """Build the seeded ``RunState`` for a graph run (M2: extracted from ``run_graph``
    so the DAG flow stays readable): multi-dataset sources + join validation, the
    feature-config unpacking, offload policy, and the zeroed stage-times map.

    The pipelines + execution params come from ``context`` (D1/F6 — no longer re-listed as
    a dozen individual params); ``features`` is the caller-normalized feature bundle.
    """
    dataset_sources, disable_ir_metrics, join_warning = _load_multi_dataset_sources(
        eval_config
    )
    state = RunState(
        dataset=dataset,
        mode=mode,
        dataset_sources=dataset_sources,
        disable_ir_metrics=disable_ir_metrics,
        join_warning=join_warning,
        retrieval_pipeline=context.retrieval_pipeline,
        asr_pipeline=context.asr_pipeline,
        text_embedding_pipeline=context.text_embedding_pipeline,
        audio_embedding_pipeline=context.audio_embedding_pipeline,
        cache_manager=context.cache_manager,
        config=eval_config,
        load_info=load_info,
        k=context.k,
        batch_size=context.batch_size,
        num_workers=context.num_workers,
        checkpoint_interval=context.checkpoint_interval,
        experiment_id=context.experiment_id,
        resume_from_checkpoint=context.resume_from_checkpoint,
        oracle_mode=context.oracle_mode,
        embedding_fusion_config=features.embedding_fusion_config,
        query_opt_config=features.query_opt_config,
        query_correction_config=features.query_correction_config,
        answer_gen_config=features.answer_gen_config,
        judge_config=features.judge_config,
        trace_limit=context.trace_limit,
        term_weights=features.term_weights,
        compute_confidence_intervals=features.compute_confidence_intervals,
        total=total,
        cb=cb,
        t_total=t_total,
        service_provider=service_provider,
        offload_after_stage=(
            service_provider is not None
            and offload_policy in ("on_finish", "on_finish_soft_cpu")
        ),
        soft_cpu_offload=(
            service_provider is not None and offload_policy == "on_finish_soft_cpu"
        ),
        refine_in_graph=any(
            _node_kind(n) in ("rerank", "mmr", "threshold") for n in stage_graph.nodes
        ),
        mmr_in_graph=any(_node_kind(n) == "mmr" for n in stage_graph.nodes),
        # A hybrid result_fusion consumes the dense + sparse arms' candidate pools, so those
        # retrievals must emit candidates (not finalize to k) even with no refine node.
        fuse_in_graph=any(
            _node_kind(n) == "result_fusion" and (n.params or {}).get("hybrid")
            for n in stage_graph.nodes
        ),
    )
    state.stage_times = {
        "asr_s": 0.0,
        "query_opt_s": 0.0,
        "correction_s": 0.0,
        "embedding_s": 0.0,
        "retrieval_s": 0.0,
    }
    # Soft-CPU offload (2c): size the provider's warm pool from config before any release.
    if state.soft_cpu_offload and hasattr(service_provider, "configure_soft_offload"):
        sr = getattr(eval_config, "service_runtime", None)
        service_provider.configure_soft_offload(
            max_warm=getattr(sr, "soft_offload_max_warm", 2),
            ttl_s=getattr(sr, "soft_offload_ttl_s", None),
        )
    return state


def _finalize_run(cache_manager: Any, experiment_id: Any) -> None:
    """Post-run housekeeping: drop the run's phased checkpoint + log cache stats."""
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
    context = EvaluationContext(
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
        features=features,
    )
    # Non-context run params (not bundled in EvaluationContext): provider + report inputs.
    _run_kwargs = dict(
        service_provider=bundle.service_provider,
        eval_config=config,
        load_info=load_info,
        graph_override=getattr(config, "graph_override", None),
    )
    results = run_graph(
        dataset,
        context,
        offload_policy="never" if _oracle_will_run else _offload_policy,
        **_run_kwargs,
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
        oracle_context = _replace(
            context, oracle_mode=True, features=oracle_features, trace_limit=0
        )
        oracle_results = run_graph(
            dataset,
            oracle_context,
            offload_policy=_offload_policy,  # last use of these pipelines → safe to offload
            **_run_kwargs,
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
