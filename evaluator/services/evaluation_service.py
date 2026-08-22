"""Evaluation orchestration service used by public API."""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from ..config import EvaluationConfig
from ..errors import EvaluatorError
from ..evaluation.results import EvaluationResults
from ..logging_config import get_logger, setup_logging
from ..pipeline import create_pipeline_from_config
from ..storage.cache import CacheManager
from ..storage.leaderboard import ExperimentStore
from ..tracking import MLflowTracker, NoOpTracker
from ..datasets import (
    load_runtime_dataset,
    validate_dataset_runtime_config,
)
from ..datasets.profiles import profile_snapshot
from ..evaluation.load_info import LoadInfo
from .model_provider import ModelServiceProvider


def _create_tracker(config: EvaluationConfig):
    if not config.tracking.enabled:
        return NoOpTracker()
    experiment_name = config.tracking.mlflow_experiment_name or config.experiment_name
    if config.tracking.backend == "mlflow":
        return MLflowTracker(
            experiment_name=experiment_name,
            tracking_uri=config.tracking.mlflow_tracking_uri,
        )
    return NoOpTracker()


def _resolve_profile_snapshot(config: EvaluationConfig) -> Dict[str, Any]:
    return profile_snapshot(config)


def _evaluate_metrics(
    config: EvaluationConfig,
    dataset,
    bundle,
    cache_manager,
    tracker,
    progress_callback=None,
    load_info=None,
):
    from ..evaluation.executor.run import run_from_bundle

    with tracker:
        tracker.log_params(
            {
                "runtime": config.to_runtime_dict(),
                "experiment": config.to_experiment_dict(),
            }
        )

        # Single execution core: the DAG executor. `config.parallel_enabled` drives the
        # executor's intra-level per-branch concurrency (read inside the DAG run path);
        # there is no separate data-parallel bypass (the multiprocess ParallelEvaluator
        # was retired; the DAG executor is the single in-process parallel path).
        metrics = run_from_bundle(
            dataset,
            bundle,
            config,
            cache_manager=cache_manager,
            progress_callback=progress_callback,
            load_info=load_info,
        )
        tracker.log_metrics(metrics)
        return metrics


def _cache_delta(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    """Compute compact cache growth deltas from cache stats snapshots."""
    before_sizes = before.get("sizes_bytes", {})
    after_sizes = after.get("sizes_bytes", {})
    before_files = before.get("file_counts", {})
    after_files = after.get("file_counts", {})
    keys = sorted(set(before_sizes.keys()) | set(after_sizes.keys()))
    return {
        "size_bytes_delta": {
            k: after_sizes.get(k, 0) - before_sizes.get(k, 0) for k in keys
        },
        "file_count_delta": {
            k: after_files.get(k, 0) - before_files.get(k, 0) for k in keys
        },
    }


def _warm_up_model_services(config: EvaluationConfig, bundle: Any, logger) -> None:
    """Eager startup: touch each configured pipeline's model so lazy construction runs now."""
    mode = config.service_runtime.startup_mode
    logger.info("service.startup_policy mode=%s", mode)
    if mode != "eager":
        return
    # Force metadata/name access so service-backed models are touched eagerly.
    touched = []
    if bundle.asr_pipeline is not None:
        touched.append(bundle.asr_pipeline.model.name())
    if bundle.text_embedding_pipeline is not None:
        touched.append(bundle.text_embedding_pipeline.model.name())
    if bundle.audio_embedding_pipeline is not None:
        touched.append(bundle.audio_embedding_pipeline.model.name())
    logger.info("service.startup_eager models=%s", touched)


def _configure_local_llm_runtime(
    config: EvaluationConfig, service_provider: ModelServiceProvider, logger
) -> None:
    """Start local LLM server when local judge/query optimization mode is enabled.

    Checked across the flat default AND every graph node's resolved override — a node-level
    override that turns on ``use_local_server`` only for one branch must not be invisible here
    (the server just wouldn't come up, and that node's resolved config would silently point at
    no local endpoint)."""
    from ..config.validation import _configured_feature_configs

    judge_local = any(
        cfg.enabled and cfg.use_local_server
        for cfg in _configured_feature_configs(config, "judge")
    )
    query_local = any(
        cfg.enabled and cfg.use_local_server
        for cfg in _configured_feature_configs(config, "query_optimization")
    )
    if not (judge_local or query_local):
        return

    llm_server = service_provider.get_llm_server(config.llm_server)
    local_api_url = llm_server.get_api_url()
    logger.info(
        "llm.local_runtime_ready backend=%s model=%s api_url=%s",
        config.llm_server.backend,
        config.llm_server.model,
        local_api_url,
    )

    if judge_local:
        config.judge.local_server_url = local_api_url
        config.judge.api_base = local_api_url
    if query_local:
        config.query_optimization.local_server_url = local_api_url
        config.query_optimization.llm_api_base = local_api_url


def _run_core(
    config: EvaluationConfig,
    *,
    cache_manager,
    service_provider=None,
    progress_callback=None,
    load_info: Optional[LoadInfo] = None,
    query_ids: Optional[Any] = None,
):
    """Build pipelines, load the dataset (+corpus/synthesis), and evaluate.

    The single execution core shared by the public API (with a provider), the CLI, and
    the webapi. Returns ``(metrics, dataset)``. The caller owns provider lifecycle,
    logging, metadata, and leaderboard ingest.
    """
    if load_info is None:
        load_info = LoadInfo()
    logger = get_logger(__name__)

    # Pre-flight chain (evaluation/validation.py): determinism seed, LLM budget,
    # embedding-space typing (config + per-node graph), optional store backends.
    from ..evaluation.validation import run_pre_flight

    run_pre_flight(config)

    # Validate the dataset config pre-graph (no data load — loading is the in-graph dataset_source
    # node now). TTS synthesis is in-graph (the tts node). Item replay (2d) rides ``load_info``
    # into the source node, which applies the query-id slice at load (corpus kept whole).
    validate_dataset_runtime_config(
        config, retrieval_required=_run_needs_retrieval(config)
    )
    if query_ids is not None:
        load_info.replay_query_ids = list(query_ids)

    # Run-start summary on the shared service path (the CLI logs the full config at debug; this
    # one INFO line makes a webapi/API run diagnosable: what ran, over how much data).
    from ..logging_config import runtime_logger

    runtime_logger.info(
        "run start: template=%s retrieval=%s fusion=%s correction=%s query_opt=%s",
        config.graph_template,
        _run_needs_retrieval(config),
        config.embedding_fusion.enabled,
        config.query_correction.enabled,
        config.query_optimization.enabled,
    )  # the dataset_source node logs "Dataset size: N" once it loads in-graph

    bundle = create_pipeline_from_config(
        config, cache_manager, service_provider=service_provider
    )
    _warm_up_model_services(config, bundle, logger)
    if service_provider is not None:
        _configure_local_llm_runtime(config, service_provider, logger)

    # Corpus embedding + index build are the in-graph corpus_embedding + vector_db nodes (run
    # inside run_from_bundle), so it is no longer done eagerly here.
    tracker = _create_tracker(config)
    metrics = _evaluate_metrics(
        config,
        None,  # the dataset is loaded in-graph by the dataset_source node (see load_info below)
        bundle,
        cache_manager,
        tracker,
        progress_callback=progress_callback,
        load_info=load_info,
    )
    # The in-graph load stashed the loaded dataset on the shared ``load_info`` dict, so the
    # caller's num_samples/metadata needs no pre-graph load.
    dataset = load_info.dataset
    return metrics, dataset


def run_evaluation(
    config: EvaluationConfig, progress_callback=None
) -> EvaluationResults:
    """Run complete evaluation lifecycle for a validated config.

    Thin wrapper over :func:`_run_core`: provider lifecycle, logging, metadata, and
    leaderboard ingest.
    """
    start_time = datetime.now()
    service_provider = ModelServiceProvider()

    try:
        # Setup errors propagate with their real types (ConfigurationError / OSError / …)
        # — the CLI + api boundaries already format them; a blanket RuntimeError wrap
        # only erased the type.
        setup_logging(
            log_dir=config.logging.log_dir,
            console_level=config.logging.get_console_level(),
            file_level=config.logging.get_file_level(),
            experiment_name=config.experiment_name,
            verbosity=config.logging.verbosity,
        )
        from ..logging_config import add_run_file_handler
        add_run_file_handler(config.output_dir, config.experiment_name)
        cache_manager = CacheManager(
            cache_dir=config.cache.cache_dir,
            enabled=config.cache.enabled,
            max_size_gb=config.cache.max_size_gb,
        )

        load_info = LoadInfo()
        cache_stats_before = (
            cache_manager.get_cache_stats() if cache_manager.enabled else {}
        )

        try:
            metrics, dataset = _run_core(
                config,
                cache_manager=cache_manager,
                service_provider=service_provider,
                progress_callback=progress_callback,
                load_info=load_info,
            )
        except EvaluatorError:
            raise  # typed errors (ConfigurationError, …) keep their type for the caller
        except Exception as e:
            raise RuntimeError(f"Evaluation failed: {e}") from e

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        metadata = {
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": duration,
            "pipeline_mode": config.graph_template,
            "dataset_profile": _resolve_profile_snapshot(config),
        }
        if hasattr(dataset, "__len__"):
            metadata["num_samples"] = len(dataset)
        if cache_manager.enabled:
            cache_stats_after = cache_manager.get_cache_stats()
            metadata["cache"] = {
                "enabled": True,
                "stats_before": cache_stats_before,
                "stats_after": cache_stats_after,
                "delta": _cache_delta(cache_stats_before, cache_stats_after),
                "load": load_info.to_metadata(),
            }
        else:
            metadata["cache"] = {"enabled": False}

        results = EvaluationResults(
            metrics=metrics,
            config=config,
            metadata=metadata,
        )
        store = ExperimentStore(
            db_path=str(Path(config.output_dir) / "leaderboard.sqlite")
        )
        run_id = store.ingest_result(results)
        results.metadata["leaderboard_run_id"] = run_id
        return results
    finally:
        service_provider.shutdown(
            offload=(config.service_runtime.offload_policy != "never")
        )


def _run_needs_retrieval(config) -> bool:
    """Whether this run retrieves (so the corpus is required). A named mode: every mode but
    ``asr_only``. A graph-only config (mode=None): derive it from the execution graph — a
    ``search`` node means retrieval (graph-first Phase 4)."""
    mode = config.graph_template
    if mode is not None:
        return str(mode) != "asr_only"
    from ..pipeline.graph.modes import build_graph_for_config

    return any(n.stage == "search" for n in build_graph_for_config(config).nodes)


def prepare_dataset(
    config: EvaluationConfig,
    *,
    retrieval_required: bool,
    cache_manager: CacheManager | None = None,
):
    """Validate + load the dataset. TTS synthesis of any missing query audio is **in-graph**
    now — the ``tts`` node (``SLOT_TTS``, present in every speech template; a free no-op, no model
    load, when nothing is missing) gap-fills, so dataset loading no longer triggers synthesis.
    Previously a pre-graph ``_synthesize_query_audio`` fallback handled the synthesis-disabled +
    audio-missing case; that's redundant now that the tts node runs regardless of the
    ``audio_synthesis.enabled`` flag. ``cache_manager`` is kept for API stability (the in-graph tts
    node caches via the run's cache manager).
    """
    validate_dataset_runtime_config(config, retrieval_required=retrieval_required)
    return load_runtime_dataset(config)


def load_dataset_and_build_index(
    config: EvaluationConfig,
    retrieval_pipeline=None,
    text_emb_pipeline=None,
    audio_emb_pipeline=None,
    cache_manager: CacheManager | None = None,
    load_info: Optional[LoadInfo] = None,
):
    """Prepare the dataset (load + synth) THEN build the corpus index — both steps,
    by design (the webapi live-preview path needs the built index)."""
    dataset = prepare_dataset(
        config, retrieval_required=(retrieval_pipeline is not None)
    )
    return build_corpus_index(
        config,
        dataset,
        retrieval_pipeline,
        text_emb_pipeline,
        audio_emb_pipeline=audio_emb_pipeline,
        cache_manager=cache_manager,
        load_info=load_info,
    )


# Re-exported so existing import paths stay stable after the F4 extraction.
from .corpus_index import build_corpus_index  # noqa: E402,F401
