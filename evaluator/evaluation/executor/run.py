"""Run entry points: ``run_graph`` + ``run_from_bundle``.

``run_graph`` is the core DAG run loop (build graph → seed run state → execute stage graph → collect
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


def _result_depth(k, source: str) -> int:
    """Retrieval depth, floored at the deepest report cutoff (@10) so the published
    ``recall@10``/``ndcg@10`` are measured rather than a shallower ``@k`` relabelled."""
    from ...metrics.ir import RETRIEVAL_DEPTH, report_depth

    depth = report_depth(k)
    if depth != int(k):
        logger.info(
            "%s=%s raised to %d: the report publishes cutoffs up to @%d",
            source, k, depth, RETRIEVAL_DEPTH,
        )
    return depth


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
    device_pool: Any = None,
    offload_policy: str = "never",
    eval_config: Any = None,
    load_info: Any = None,
    graph_override: Any = None,
) -> RunResults:
    """Run the evaluation by dispatching the stage-graph DAG node by node (the graph, not
    fixed phases, is the source of truth).

    Args:
        dataset: QueryDataset instance containing audio samples and ground truth.
        context: EvaluationContext bundling the pipelines + execution params (D1/F5 — the
            sole entry contract: retrieval/asr/text-emb/audio-emb pipelines, cache_manager,
            k, batch_size, trace_limit, num_workers, checkpoint_interval, experiment_id,
            resume_from_checkpoint, progress_callback, oracle_mode, and ``features``).
        service_provider: Optional model service provider (offload coordination).
        device_pool: Optional GPUPool (memory-aware allocation + LRU eviction) — reaches
            per-node model-override builds too (`_build_node_pipeline`), not just the
            shared global pipeline built before this call.
        offload_policy: "never" | "on_finish" | "on_finish_soft_cpu" — free each stage's
            model after last use ("on_finish_soft_cpu" parks it warm on CPU instead).
        eval_config: The EvaluationConfig (mode detection, multi-dataset sources, provenance).
        load_info: Optional model load metadata for the report.
        graph_override: Optional explicit stage-graph spec replacing the default for the mode.

    Returns:
        RunResults — the flat metric dict (pipeline_mode, WER/CER, MRR/Recall@k/NDCG@k, …,
        total_samples, duration_seconds) the report/leaderboard consume.

    Raises:
        ValueError: If neither audio_embedding_pipeline nor asr_pipeline provided.
    """
    features = context.features or RunFeatures()
    mode, stage_graph = _resolve_mode_and_graph(
        context, features, eval_config, graph_override
    )

    # The dataset is loaded in-graph (the dataset_source node) — on the standard path ``dataset``
    # is None here and the node sets ``state.total`` when it loads; a pre-loaded dataset (oracle
    # re-run / direct callers) keeps the count up front.
    _have_ds = dataset is not None
    logger.info(
        "Batch size: %s, k: %s%s",
        context.batch_size, context.k,
        f", dataset size: {len(dataset)}" if _have_ds else " (dataset loads in-graph)",
    )
    # Pre-flight (M3): fail a typo'd/unregistered node type before any heavy work; a
    # model node without a device mapping fails here too (was an import-time assert).
    validate_graph_handlers(stage_graph)
    from .parallel import assert_model_nodes_are_device_managed

    assert_model_nodes_are_device_managed()

    _total = len(dataset) if _have_ds else 0
    _cb = context.progress_callback or (lambda *_: None)
    _cb("init", 0, _total, f"Starting {mode} evaluation")

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
        device_pool=device_pool,
        offload_policy=offload_policy,
    )

    _dispatch_execution(state, stage_graph, eval_config, features)

    _finalize_run(context.cache_manager, context.experiment_id)
    result = _collect_final_result(state, stage_graph)
    _cb("done", _total, _total, "Evaluation complete")
    return result


def _collect_final_result(state: "RunState", stage_graph) -> RunResults:
    """Assemble the run's returned result from every `finalize` node's own published
    ``run_report`` (R4/multi-variant — each finalize node builds its own report from its own
    bound producers; see `_stage_finalize`). Exactly one → that report, flat, byte-identical to
    the pre-R4 single-shared-`state.results` behavior. N > 1 (a multi-variant graph) →
    ``{"variants": {node_id: report, ...}, "latency": ..., "node_latency": ...}``. Zero (no
    `finalize` node in the graph, e.g. a structural-only run) → unchanged fallback to the
    shared `state.results`."""
    finalize_ids = [n.id for n in stage_graph.nodes if _node_kind(n) == "finalize"]
    reports = [
        (nid, state.ctx.get(nid, "run_report"))
        for nid in finalize_ids
        if state.ctx.has(nid, "run_report")
    ]
    # Branch fail-fast: a failed branch's finalize never published a run_report — name the
    # casualties in the result (and in one end-of-run warning) instead of dropping them
    # silently next to the healthy variants.
    failed = dict(getattr(state, "failed_nodes", {}) or {})
    skipped = dict(getattr(state, "skipped_nodes", {}) or {})
    if failed:
        summary = "; ".join(f"{nid} ({err})" for nid, err in sorted(failed.items()))
        logger.warning(
            "%d node(s) failed: %s — %d downstream node(s) skipped",
            len(failed), summary, len(skipped),
        )
    # G5: traces/judge are consolidated into report['traces']/['judge'] during finalize; drop
    # the duplicate top-level keys at the output boundary (sinks read them during the run).
    if not reports:
        drop_mirrored_top_level_keys(state.results)
        if failed and isinstance(state.results, dict):
            state.results["failed_nodes"] = failed
            state.results["skipped_nodes"] = skipped or None
        return state.results
    for _, report in reports:
        drop_mirrored_top_level_keys(report)
    if len(reports) == 1 and not failed:
        return reports[0][1]
    result: RunResults = {
        "variants": dict(reports),
        "latency": dict(state.stage_times),
        "node_latency": dict(state.node_times),
    }
    if failed:
        result["failed_nodes"] = failed
        result["skipped_nodes"] = skipped or None
    return result


def _resolve_mode_and_graph(
    context: "EvaluationContext",
    features: "RunFeatures",
    eval_config: Any,
    graph_override: Any,
):
    """Determine the template label + build the run stage graph (the source of truth for
    what runs). The config's graph template wins — cross-modal audio_emb builds a text
    pipeline for the corpus, which pipeline-presence detection would misread as audio_text
    fusion."""
    configured_mode = None
    if eval_config is not None and getattr(eval_config, "model", None) is not None:
        from ...pipeline.graph.modes import resolve_template_label

        # a back-compat template reference wins; else derive the label from the explicit graph's
        # node kinds (audio_emb vs audio_text hinges on corpus_embedding vs text_embedding),
        # which presence-detection alone can't tell apart.
        configured_mode = resolve_template_label(eval_config)
    mode = detect_graph_template(
        context.retrieval_pipeline,
        context.asr_pipeline,
        context.text_embedding_pipeline,
        context.audio_embedding_pipeline,
        configured_mode=configured_mode,
    )
    logger.info("Evaluation template: %s (DAG)", GRAPH_TEMPLATES.get(mode, mode))
    stage_graph = build_run_graph(
        mode,
        graph_override=graph_override,
        embedding_fusion_config=features.embedding_fusion_config,
        query_opt_config=features.query_opt_config,
        query_correction_config=features.query_correction_config,
        retrieval_pipeline=context.retrieval_pipeline,
        eval_config=eval_config,
        trace_limit=context.trace_limit,
    )
    stage_levels = [
        [node.id for node in level] for level in stage_graph.topological_levels()
    ]
    node_logger.info("Execution DAG mode=%s levels=%s", mode, stage_levels)
    return mode, stage_graph


def _dispatch_execution(
    state: "RunState", stage_graph: Any, eval_config: Any, features: "RunFeatures"
) -> None:
    """DAG-driven execution: the stage graph drives what runs and in what order. With
    ``streaming.window_size`` set, run the query side window-by-window (3a) — same
    RunState, bounded memory; else the whole-dataset pass."""
    _window = getattr(getattr(eval_config, "streaming", None), "window_size", None)
    if _window:
        from .streaming import execute_windowed

        execute_windowed(state, stage_graph, int(_window))
    else:
        _execute_stage_graph(state, stage_graph, features.query_opt_config)


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
    device_pool: Any = None,
    offload_policy: str = "never",
) -> RunState:
    """Build the seeded ``RunState`` for a graph run (M2: extracted from ``run_graph``
    so the DAG flow stays readable): multi-dataset sources + join validation, the
    feature-config unpacking, offload policy, and the zeroed stage-times map.

    The pipelines + execution params come from ``context`` (D1/F6 — no longer re-listed as
    a dozen individual params); ``features`` is the caller-normalized feature bundle.
    """
    # The dataset is loaded in-graph (the dataset_source node) — when ``dataset`` is None here,
    # that node loads the sources + the join gate. The only callers that still pass a pre-loaded
    # dataset (the oracle re-run, direct ``run_graph`` callers, tests) get the sources loaded here.
    if dataset is not None:
        from ...datasets.runtime import load_dataset_sources

        dataset_sources, disable_ir_metrics, join_warning = load_dataset_sources(eval_config)
    else:
        dataset_sources, disable_ir_metrics, join_warning = {}, False, ""
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
        metric_allowlist=features.metric_allowlist,
        variant_rollup=features.variant_rollup,
        total=total,
        cb=cb,
        t_total=t_total,
        service_provider=service_provider,
        device_pool=device_pool,
        offload_after_stage=(
            service_provider is not None
            and offload_policy in ("on_finish", "on_finish_soft_cpu")
        ),
        soft_cpu_offload=(
            service_provider is not None and offload_policy == "on_finish_soft_cpu"
        ),
        **_graph_shape_flags(stage_graph),
    )
    # resolve every feature node's effective config ONCE (global ⊕ node params),
    # so handlers look theirs up instead of overlaying an allowlist at run time.
    from ..node_config import resolve_graph_node_configs

    state.node_configs = resolve_graph_node_configs(state, stage_graph)
    state.stage_times = {
        "asr_s": 0.0,
        "query_opt_s": 0.0,
        "correction_s": 0.0,
        "embedding_s": 0.0,
        "retrieval_s": 0.0,
    }
    _configure_offload(state, service_provider, eval_config)
    return state


def _graph_shape_flags(stage_graph: Any) -> Dict[str, bool]:
    """Handler-behavior flags derived from the graph's node kinds."""
    return {
        "refine_in_graph": any(
            _node_kind(n) in ("rerank", "mmr", "threshold") for n in stage_graph.nodes
        ),
        "mmr_in_graph": any(_node_kind(n) == "mmr" for n in stage_graph.nodes),
        # A hybrid result_fusion consumes the dense + sparse arms' candidate pools, so those
        # retrievals must emit candidates (not finalize to k) even with no refine node.
        "fuse_in_graph": any(
            _node_kind(n) == "result_fusion" and (n.params or {}).get("hybrid")
            for n in stage_graph.nodes
        ),
    }


def _configure_offload(state: RunState, service_provider: Any, eval_config: Any) -> None:
    """Soft-CPU offload (2c): size the provider's warm pool from config before any release."""
    if state.soft_cpu_offload and hasattr(service_provider, "configure_soft_offload"):
        sr = getattr(eval_config, "service_runtime", None)
        service_provider.configure_soft_offload(
            max_warm=getattr(sr, "soft_offload_max_warm", 2),
            ttl_s=getattr(sr, "soft_offload_ttl_s", None),
        )


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
    dataset: Optional[QueryDataset],
    bundle: "PipelineBundle",
    config: "EvaluationConfig",
    *,
    cache_manager: Optional[CacheManager] = None,
    progress_callback: Optional[Callable[[str, int, int, str], None]] = None,
    load_info: Optional[Dict[str, Any]] = None,
) -> RunResults:
    """Build ``RunFeatures`` + ``EvaluationContext`` from the bundle + config, then run
    :func:`run_graph`.

    Fusion, judge, answer generation, and query optimization are enabled only when their
    respective ``config.*.enabled`` flag is True; call ``run_graph`` directly to force any
    of them on regardless of config flags.
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
        metric_allowlist=getattr(config, "metrics", None),
        variant_rollup=getattr(config, "variant_rollup", "mean"),
    )
    # Per-stage model offload: free each stage's model after its last use.
    _offload_policy = getattr(
        getattr(config, "service_runtime", None), "offload_policy", "never"
    )
    context = EvaluationContext(
        retrieval_pipeline=bundle.retrieval_pipeline,
        asr_pipeline=bundle.asr_pipeline,
        text_embedding_pipeline=bundle.text_embedding_pipeline,
        audio_embedding_pipeline=bundle.audio_embedding_pipeline,
        cache_manager=cache_manager,
        k=_result_depth(config.vector_db.k, "vector_db.k"),
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
        device_pool=bundle.device_pool,
        eval_config=config,
        load_info=load_info,
        graph_override=getattr(config, "graph_override", None),
    )
    results = run_graph(
        dataset,
        context,
        offload_policy=_offload_policy,
        **_run_kwargs,
    )

    return results
