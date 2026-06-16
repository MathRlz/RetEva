"""Evaluation orchestration service used by public API."""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from ..config import EvaluationConfig
from ..evaluation.results import EvaluationResults
from ..logging_config import get_logger, setup_logging
from ..pipeline import create_pipeline_from_config
from ..storage.cache import CacheManager
from ..storage.leaderboard import ExperimentStore
from ..tracking import MLflowTracker, NoOpTracker
from ..datasets import (
    load_runtime_dataset,
    resolve_dataset_profile,
    validate_dataset_runtime_config,
)
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
    profile = resolve_dataset_profile(
        dataset_name=config.data.dataset_name,
        dataset_type=config.data.dataset_type,
    )
    return {
        "name": profile.name,
        "dataset_type": str(profile.dataset_type),
        "requires_audio": profile.requires_audio,
        "requires_text": profile.requires_text,
        "supports_generation": profile.supports_generation,
        "evaluation_mode": profile.evaluation_mode,
        "recommended_pipeline_modes": list(profile.recommended_pipeline_modes),
        "pipeline_mode_supported": profile.supports_pipeline_mode(
            str(config.model.pipeline_mode)
        ),
    }


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


def _apply_startup_policy(config: EvaluationConfig, bundle: Any, logger) -> None:
    """Apply startup policy for model services."""
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
    """Start local LLM server when local judge/query optimization mode is enabled."""
    judge_local = bool(config.judge.enabled and config.judge.use_local_server)
    query_local = bool(
        config.query_optimization.enabled and config.query_optimization.use_local_server
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
    load_info: Optional[Dict[str, Any]] = None,
):
    """Build pipelines, load the dataset (+corpus/synthesis), and evaluate.

    The single execution core shared by the public API (with a provider), the CLI, and
    the webapi. Returns ``(metrics, dataset)``. The caller owns provider lifecycle,
    logging, metadata, and leaderboard ingest.
    """
    if load_info is None:
        load_info = {}
    logger = get_logger(__name__)

    # Pre-flight chain (evaluation/validation.py): determinism seed, LLM budget,
    # embedding-space typing (config + per-node graph), optional store backends.
    from ..evaluation.validation import run_pre_flight

    run_pre_flight(config)

    # Prepare the dataset (load + TTS-synthesize missing query audio) BEFORE building the
    # model pipelines, so a TTS model is loaded and fully offloaded before the ASR/embedding
    # models are constructed — they are never co-resident (which avoids the native crash a
    # TTS runtime can inflict on a subsequently-built embedder).
    dataset = prepare_dataset(
        config,
        retrieval_required=_mode_needs_retrieval(config.model.pipeline_mode),
        cache_manager=cache_manager,
    )

    bundle = create_pipeline_from_config(
        config, cache_manager, service_provider=service_provider
    )
    _apply_startup_policy(config, bundle, logger)
    if service_provider is not None:
        _configure_local_llm_runtime(config, service_provider, logger)

    # Corpus embedding + index build is now the in-graph ``corpus_index`` node (runs
    # inside run_from_bundle), so it is no longer done eagerly here.
    tracker = _create_tracker(config)
    metrics = _evaluate_metrics(
        config,
        dataset,
        bundle,
        cache_manager,
        tracker,
        progress_callback=progress_callback,
        load_info=load_info,
    )
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
        try:
            setup_logging(
                log_dir=config.logging.log_dir,
                console_level=config.logging.get_console_level(),
                file_level=config.logging.get_file_level(),
                experiment_name=config.experiment_name,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to setup logging: {e}") from e

        try:
            cache_manager = CacheManager(
                cache_dir=config.cache.cache_dir,
                enabled=config.cache.enabled,
                max_size_gb=config.cache.max_size_gb,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to create cache manager: {e}") from e

        load_info: Dict[str, Any] = {}
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
        except Exception as e:
            raise RuntimeError(f"Evaluation failed: {e}") from e

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        metadata = {
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": duration,
            "pipeline_mode": config.model.pipeline_mode,
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
                "load": load_info,
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


def _mode_needs_retrieval(mode) -> bool:
    """Every pipeline mode retrieves except ``asr_only``."""
    return str(mode) != "asr_only"


def prepare_dataset(
    config: EvaluationConfig,
    *,
    retrieval_required: bool,
    cache_manager: CacheManager | None = None,
):
    """Validate + load the dataset and synthesize any missing query audio.

    Runs before the model pipelines are built so a TTS model is loaded and offloaded
    before the ASR/embedding models are constructed (no co-resident native runtimes).
    Synthesized audio is cached via ``cache_manager`` (so ``--no-cache``/``--clear-cache``
    apply) when supplied.
    """
    logger = get_logger(__name__)
    validate_dataset_runtime_config(config, retrieval_required=retrieval_required)
    dataset = load_runtime_dataset(config)
    _synthesize_query_audio(config, dataset, logger, cache_manager)
    return dataset


def _synthesize_query_audio(
    config: EvaluationConfig, dataset, logger, cache_manager=None
) -> None:
    """Synthesize audio for questions lacking ``audio_path`` (when enabled or needed).

    When ``audio_synthesis.enabled`` the in-graph ``tts`` node performs synthesis, so
    this pre-graph path only covers the implicit fallback: questions missing audio with
    synthesis not explicitly enabled.
    """
    if config.audio_synthesis.enabled:
        return  # handled by the in-graph tts node
    questions_missing_audio = (
        hasattr(dataset, "questions")
        and dataset.questions
        and any(not getattr(q, "audio_path", None) for q in dataset.questions)
    )
    if not questions_missing_audio:
        return
    if questions_missing_audio and not config.audio_synthesis.enabled:
        logger.info("Audio missing for some questions — auto-synthesizing with TTS")
    else:
        logger.info(
            "Audio synthesis enabled - checking for questions needing synthesis"
        )
    if not (hasattr(dataset, "questions") and dataset.questions):
        logger.warning(
            "Dataset does not have 'questions' attribute, TTS synthesis skipped"
        )
        return
    from ..pipeline.audio.prepare import synthesize_missing_query_audio

    logger.info(
        "STAGE prepare_dataset: TTS synthesis START (provider=%s)",
        config.audio_synthesis.provider,
    )
    synthesize_missing_query_audio(
        dataset.questions,
        config.audio_synthesis,
        log=logger,
        cache_manager=cache_manager,
    )
    logger.info("STAGE prepare_dataset: TTS synthesis DONE")


def load_dataset(
    config: EvaluationConfig,
    retrieval_pipeline=None,
    text_emb_pipeline=None,
    audio_emb_pipeline=None,
    cache_manager: CacheManager | None = None,
    load_info: Dict[str, Any] | None = None,
):
    """Back-compat: prepare the dataset (load + synth) then build the corpus index."""
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
from .matrix import (  # noqa: E402,F401
    _apply_setup_overrides,
    _build_comparison_bundle,
    run_evaluation_matrix,
)
