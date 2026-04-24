"""Evaluation orchestration service used by public API."""

from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from ..config import EvaluationConfig
from ..config.types import enum_to_str
from ..errors import ConfigurationError
from ..evaluation.results import EvaluationResults
from ..logging_config import get_logger, setup_logging
from ..pipeline import create_pipeline_from_config
from ..storage.cache import CacheManager
from ..storage.leaderboard import ExperimentStore
from ..storage.cache_keys import (
    dataset_fingerprint,
    manifest_fingerprint,
    model_fingerprint,
    preprocessing_fingerprint,
    retrieval_fingerprint,
    vector_db_manifest_key,
)
from ..tracking import MLflowTracker, NoOpTracker
from ..datasets import load_runtime_dataset, resolve_dataset_profile, validate_dataset_runtime_config
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


def _resolve_dataset_identity(config: EvaluationConfig) -> tuple[str, Dict[str, Any]]:
    source: Dict[str, Any]
    if config.data.prepared_dataset_dir:
        dataset_dir = Path(config.data.prepared_dataset_dir).resolve()
        source = {"prepared_dataset_dir": str(dataset_dir)}
        return f"prepared:{dataset_dir}", source
    if config.data.questions_path:
        questions_path = Path(config.data.questions_path).resolve()
        corpus_path = Path(config.data.corpus_path).resolve() if config.data.corpus_path else None
        source = {
            "questions_path": str(questions_path),
            "corpus_path": str(corpus_path) if corpus_path else None,
        }
        return f"questions:{questions_path}|corpus:{corpus_path}", source
    return "unknown", {}


def _corpus_signature(corpus: List[Any]) -> str:
    normalized = []
    for doc in corpus:
        if isinstance(doc, dict):
            normalized.append(
                {
                    "doc_id": doc.get("doc_id"),
                    "text": doc.get("text"),
                }
            )
        else:
            normalized.append(str(doc))
    return manifest_fingerprint({"corpus": normalized})


def _vector_db_cache_key(config: EvaluationConfig, retrieval_pipeline: Any, corpus: List[Any]) -> str:
    dataset_identity, source = _resolve_dataset_identity(config)
    dataset_fp = dataset_fingerprint(
        dataset_identity=dataset_identity,
        trace_limit=config.data.trace_limit,
        source={
            **source,
            "corpus_signature": _corpus_signature(corpus),
            "corpus_size": len(corpus),
        },
    )
    model_fp = model_fingerprint(
        model_name=config.model.text_emb_model_name or config.model.text_emb_model_type or "unknown",
        model_type=config.model.text_emb_model_type,
        inference={"device": config.model.text_emb_device},
    )
    strategy = {
        "mode": config.vector_db.retrieval_mode,
        "hybrid_dense_weight": config.vector_db.hybrid_dense_weight,
        "hybrid_fusion_method": config.vector_db.hybrid_fusion_method,
        "rrf_k": config.vector_db.rrf_k,
        "bm25_k1": config.vector_db.bm25_k1,
        "bm25_b": config.vector_db.bm25_b,
        "reranker_mode": config.vector_db.reranker_mode,
        "reranker_enabled": config.vector_db.reranker_enabled,
        "reranker_model": config.vector_db.reranker_model,
        "reranker_weight": config.vector_db.reranker_weight,
        "use_mmr": config.vector_db.use_mmr,
        "mmr_lambda": config.vector_db.mmr_lambda,
        "min_similarity_threshold": config.vector_db.min_similarity_threshold,
        "distance_metric": config.vector_db.distance_metric,
    }
    retrieval_fp = retrieval_fingerprint(
        vector_store_type=enum_to_str(config.vector_db.type),
        retrieval_strategy=strategy,
    )
    preprocess_fp = preprocessing_fingerprint(
        {
            "pipeline_mode": config.model.pipeline_mode,
            "query_optimization_enabled": config.query_optimization.enabled,
        }
    )
    return vector_db_manifest_key(
        dataset_fp=dataset_fp,
        model_fp=model_fp,
        retrieval_fp=retrieval_fp,
        preprocessing_fp=preprocess_fp,
    )


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


def _evaluate_metrics(config: EvaluationConfig, dataset, bundle, cache_manager, tracker):
    from ..evaluation.phased import evaluate_from_bundle

    with tracker:
        tracker.log_params(
            {
                "runtime": config.to_runtime_dict(),
                "experiment": config.to_experiment_dict(),
            }
        )

        if config.parallel_enabled:
            from ..parallel import ParallelEvaluator
            from ..config import get_available_gpu_count

            num_gpus = get_available_gpu_count()
            if num_gpus > 1 or config.num_parallel_workers > 1:
                parallel_evaluator = ParallelEvaluator(
                    config=config,
                    num_workers=config.num_parallel_workers or None,
                )
                metrics = parallel_evaluator.evaluate_parallel(
                    dataset=dataset,
                    k=config.vector_db.k,
                    batch_size=config.data.batch_size,
                    trace_limit=config.data.trace_limit,
                )
            else:
                metrics = evaluate_from_bundle(
                    dataset, bundle, config, cache_manager=cache_manager,
                )
        else:
            metrics = evaluate_from_bundle(
                dataset, bundle, config, cache_manager=cache_manager,
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
        "size_bytes_delta": {k: after_sizes.get(k, 0) - before_sizes.get(k, 0) for k in keys},
        "file_count_delta": {k: after_files.get(k, 0) - before_files.get(k, 0) for k in keys},
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


def run_evaluation(config: EvaluationConfig) -> EvaluationResults:
    """Run complete evaluation lifecycle for a validated config."""
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

        tracker = _create_tracker(config)

        try:
            cache_manager = CacheManager(
                cache_dir=config.cache.cache_dir,
                enabled=config.cache.enabled,
                max_size_gb=config.cache.max_size_gb,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to create cache manager: {e}") from e

        try:
            bundle = create_pipeline_from_config(
                config,
                cache_manager,
                service_provider=service_provider,
            )
            logger = get_logger(__name__)
            _apply_startup_policy(config, bundle, logger)
            _configure_local_llm_runtime(config, service_provider, logger)
        except Exception as e:
            raise RuntimeError(f"Failed to create pipelines: {e}") from e

        load_info: Dict[str, Any] = {}
        cache_stats_before = cache_manager.get_cache_stats() if cache_manager.enabled else {}

        try:
            dataset = load_dataset(
                config,
                bundle.retrieval_pipeline,
                bundle.text_embedding_pipeline,
                cache_manager=cache_manager,
                load_info=load_info,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset: {e}") from e

        try:
            metrics = _evaluate_metrics(config, dataset, bundle, cache_manager, tracker)
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
        store = ExperimentStore(db_path=str(Path(config.output_dir) / "leaderboard.sqlite"))
        run_id = store.ingest_result(results)
        results.metadata["leaderboard_run_id"] = run_id
        return results
    finally:
        service_provider.shutdown(
            offload=(config.service_runtime.offload_policy != "never")
        )


def _apply_setup_overrides(config: EvaluationConfig, overrides: Dict[str, Any]) -> EvaluationConfig:
    """Apply setup overrides to a config copy."""
    nested_sections = {
        "cache",
        "logging",
        "model",
        "data",
        "audio_synthesis",
        "augmentation",
        "llm_server",
        "judge",
        "query_optimization",
        "embedding_fusion",
        "vector_db",
        "tracking",
        "device_pool",
        "service_runtime",
    }
    section_aliases = {"vector": "vector_db"}
    top_level_fields = {
        "experiment_name",
        "output_dir",
        "checkpoint_enabled",
        "checkpoint_interval",
        "resume_from_checkpoint",
        "parallel_enabled",
        "num_parallel_workers",
    }

    for key, value in overrides.items():
        if key in top_level_fields:
            setattr(config, key, value)
            continue
        if key in nested_sections and isinstance(value, dict):
            section = getattr(config, key)
            for sub_key, sub_value in value.items():
                setattr(section, sub_key, sub_value)
            continue
        matched_section = None
        sub_key = None
        for candidate in sorted(nested_sections, key=len, reverse=True):
            prefix = f"{candidate}_"
            if key.startswith(prefix):
                matched_section = candidate
                sub_key = key[len(prefix):]
                break
        if matched_section is None:
            for alias, target in section_aliases.items():
                prefix = f"{alias}_"
                if key.startswith(prefix):
                    matched_section = target
                    sub_key = key[len(prefix):]
                    break
        if matched_section is not None and sub_key:
            section = getattr(config, matched_section)
            setattr(section, sub_key, value)
            continue
        raise ConfigurationError(f"Unsupported setup override key: {key}")

    return config


def run_evaluation_matrix(
    base_config: EvaluationConfig,
    test_setups: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Run multiple test setups on the same dataset and return aggregated outputs."""
    runs: List[Dict[str, Any]] = []

    for idx, setup in enumerate(test_setups):
        setup_id = setup.get("setup_id") or setup.get("name") or f"setup_{idx + 1:03d}"
        overrides = setup.get("overrides", {})
        if not isinstance(overrides, dict):
            raise ConfigurationError(
                f"Setup '{setup_id}' must provide dict overrides, got {type(overrides).__name__}"
            )

        config_variant = deepcopy(base_config)
        config_variant = _apply_setup_overrides(config_variant, overrides)
        if "experiment_name" not in overrides:
            config_variant.experiment_name = f"{base_config.experiment_name}__{setup_id}"

        result = run_evaluation(config_variant)
        runs.append(
            {
                "setup_id": setup_id,
                "result": result.to_dict(include_config=True),
            }
        )

    comparison = _build_comparison_bundle(runs)

    return {
        "base_experiment_name": base_config.experiment_name,
        "num_setups": len(test_setups),
        "runs": runs,
        "comparison": comparison,
    }


def _build_comparison_bundle(runs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Build a stable comparison artifact (baseline deltas + leaderboard)."""
    if not runs:
        return {"baseline_setup_id": None, "metric_deltas": [], "leaderboard": []}

    baseline_setup_id = str(runs[0]["setup_id"])
    baseline_metrics = runs[0]["result"]

    def numeric_metrics(result_dict: Dict[str, Any]) -> Dict[str, float]:
        values: Dict[str, float] = {}
        for k, v in result_dict.items():
            if k.startswith("_") or isinstance(v, bool):
                continue
            if isinstance(v, (int, float)):
                values[k] = float(v)
        return values

    baseline_numeric = numeric_metrics(baseline_metrics)
    metric_deltas: List[Dict[str, Any]] = []
    leaderboard_rows: List[Dict[str, Any]] = []

    for row in runs:
        setup_id = str(row["setup_id"])
        current_numeric = numeric_metrics(row["result"])
        shared = sorted(set(baseline_numeric).intersection(current_numeric))
        deltas = {
            metric: current_numeric[metric] - baseline_numeric[metric]
            for metric in shared
        }
        metric_deltas.append(
            {
                "setup_id": setup_id,
                "deltas_vs_baseline": deltas,
            }
        )
        leaderboard_rows.append(
            {
                "setup_id": setup_id,
                "metrics": current_numeric,
            }
        )

    ranking_metric = "MRR"
    if not any(ranking_metric in row["metrics"] for row in leaderboard_rows):
        metric_names: set[str] = set()
        for row in leaderboard_rows:
            metric_names.update(row["metrics"].keys())
        ranking_metric = sorted(metric_names)[0] if metric_names else "MRR"

    leaderboard = sorted(
        leaderboard_rows,
        key=lambda r: r["metrics"].get(ranking_metric, float("-inf")),
        reverse=True,
    )

    return {
        "baseline_setup_id": baseline_setup_id,
        "ranking_metric": ranking_metric,
        "metric_deltas": metric_deltas,
        "leaderboard": leaderboard,
    }


def load_dataset(
    config: EvaluationConfig,
    retrieval_pipeline,
    text_emb_pipeline,
    cache_manager: CacheManager | None = None,
    load_info: Dict[str, Any] | None = None,
):
    """Load dataset and build retrieval index if needed."""
    from ..pipeline.audio.synthesis import AudioSynthesizer

    logger = get_logger(__name__)

    validate_dataset_runtime_config(
        config,
        retrieval_required=(retrieval_pipeline is not None),
    )
    dataset = load_runtime_dataset(config)

    questions_missing_audio = (
        hasattr(dataset, "questions")
        and dataset.questions
        and any(not getattr(q, "audio_path", None) for q in dataset.questions)
    )
    if config.audio_synthesis.enabled or questions_missing_audio:
        if questions_missing_audio and not config.audio_synthesis.enabled:
            logger.info("Audio missing for some questions — auto-synthesizing with TTS")
        else:
            logger.info("Audio synthesis enabled - checking for questions needing synthesis")
        if hasattr(dataset, "questions") and dataset.questions:
            synthesizer = AudioSynthesizer(config.audio_synthesis)
            questions_to_synthesize = [
                q for q in dataset.questions if not hasattr(q, "audio_path") or q.audio_path is None
            ]
            if questions_to_synthesize:
                logger.info(f"Synthesizing audio for {len(questions_to_synthesize)} questions...")
                for i, question in enumerate(questions_to_synthesize):
                    if not hasattr(question, "question_text") or not question.question_text:
                        logger.warning(
                            f"Question {question.question_id} has no text, skipping synthesis"
                        )
                        continue
                    output_path = None
                    if config.audio_synthesis.output_dir:
                        import os

                        os.makedirs(config.audio_synthesis.output_dir, exist_ok=True)
                        output_path = os.path.join(
                            config.audio_synthesis.output_dir,
                            f"{question.question_id}.wav",
                        )
                    try:
                        synthesizer.synthesize(question.question_text, output_path=output_path)
                        question.audio_path = output_path
                        if (i + 1) % 10 == 0:
                            logger.info(
                                f"Synthesized {i + 1}/{len(questions_to_synthesize)} questions"
                            )
                    except Exception as e:
                        logger.error(
                            f"Failed to synthesize audio for question {question.question_id}: {e}"
                        )
                logger.info(
                    f"Audio synthesis complete for {len(questions_to_synthesize)} questions"
                )
            else:
                logger.info("All questions already have audio, no synthesis needed")
        else:
            logger.warning("Dataset does not have 'questions' attribute, TTS synthesis skipped")

    if retrieval_pipeline is not None and hasattr(dataset, "get_corpus"):
        corpus = dataset.get_corpus()
        if not corpus:
            raise ConfigurationError(
                "Retrieval mode requires non-empty corpus. Set data.corpus_path "
                "to JSON/JSONL corpus file with doc_id/text fields."
            )
        if corpus and text_emb_pipeline is not None:
            corpus_texts = [doc.get("text", str(doc)) for doc in corpus]
            can_use_vector_cache = (
                cache_manager is not None and config.cache.enabled and config.cache.cache_vector_db
            )
            vector_cache_key = (
                _vector_db_cache_key(config, retrieval_pipeline, corpus)
                if can_use_vector_cache
                else None
            )
            if can_use_vector_cache and vector_cache_key:
                cached_db = cache_manager.get_vector_db(vector_cache_key)
                if cached_db is not None:
                    vectors, payloads = cached_db
                    logger.info(
                        "Loaded cached retrieval index artifacts: vectors=%d",
                        len(vectors),
                    )
                    retrieval_pipeline.build_index(embeddings=vectors, metadata=payloads)
                    if load_info is not None:
                        load_info["vector_cache_hit"] = True
                        load_info["vector_cache_key"] = vector_cache_key
                        load_info["corpus_size"] = len(corpus)
                    return dataset
                if load_info is not None:
                    load_info["vector_cache_hit"] = False
                    load_info["vector_cache_key"] = vector_cache_key
                    load_info["corpus_size"] = len(corpus)

            corpus_embeddings = text_emb_pipeline.process_batch(corpus_texts, show_progress=True)
            vectors = np.array(corpus_embeddings)
            retrieval_pipeline.build_index(embeddings=vectors, metadata=corpus)
            if can_use_vector_cache and vector_cache_key:
                cache_manager.set_vector_db(vector_cache_key, vectors, corpus)
                logger.info("Cached retrieval index artifacts for reuse")
                if load_info is not None:
                    load_info["vector_cache_written"] = True
                    load_info["vector_cache_key"] = vector_cache_key
                    load_info["corpus_size"] = len(corpus)
            elif load_info is not None:
                load_info["vector_cache_hit"] = False
                load_info["corpus_size"] = len(corpus)
    return dataset
