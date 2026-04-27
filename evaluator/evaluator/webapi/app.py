"""FastAPI application exposing evaluator runtime as HTTP backend."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from pydantic import BaseModel, Field

from evaluator import ConfigurationError, EvaluationConfig, list_presets, run_evaluation, run_evaluation_matrix
from evaluator.config.types import DatasetType, PipelineMode, VectorDBType
from evaluator.datasets import (
    list_known_dataset_names,
    resolve_dataset_profile,
    validate_dataset_runtime_config,
)
from evaluator.pipeline import build_stage_graph, create_pipeline_from_config
from evaluator.pipeline.factory import check_backend_dependencies
from evaluator.services.evaluation_service import load_dataset
from evaluator.services import ModelServiceProvider
from evaluator.storage import ExperimentStore
from evaluator.storage.cache import CacheManager


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Request models ──────────────────────────────────────────────────


class EvaluationJobRequest(BaseModel):
    """Payload for single evaluation job creation."""

    config: Dict[str, Any] = Field(default_factory=dict, description="EvaluationConfig dict (flat or nested)")
    auto_devices: bool = Field(True, description="Auto-configure device assignments based on hardware")


class MatrixJobRequest(BaseModel):
    """Payload for matrix evaluation job creation."""

    base_config: Dict[str, Any] = Field(default_factory=dict, description="Base EvaluationConfig dict")
    test_setups: List[Dict[str, Any]] = Field(default_factory=list, description="List of override dicts per setup")
    auto_devices: bool = True


class ConfigCreateRequest(BaseModel):
    """Payload for config creator endpoint."""

    preset_name: Optional[str] = Field(None, description="Preset to start from (e.g. 'fast_dev')")
    config_patch: Dict[str, Any] = Field(default_factory=dict, description="Nested patch dict to merge over preset")
    auto_devices: bool = True


class LiveQueryRequest(BaseModel):
    """Payload for ad-hoc live retrieval query."""

    config: Dict[str, Any] = Field(default_factory=dict, description="EvaluationConfig dict for pipeline setup")
    query_text: str = Field("", description="Query text to search")
    k: int = Field(5, description="Number of results to return")
    auto_devices: bool = True


# ── Response models ─────────────────────────────────────────────────


class HealthResponse(BaseModel):
    status: str = "ok"


class JobSubmitResponse(BaseModel):
    job_id: str


class JobStatusResponse(BaseModel):
    job_id: str
    job_type: str
    status: str
    submitted_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error: Optional[str] = None


class ErrorResponse(BaseModel):
    detail: str


@dataclass
class JobRecord:
    """In-memory job state for async API runs."""

    job_id: str
    job_type: str
    status: str = "queued"
    submitted_at: str = field(default_factory=_utc_now)
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    config_snapshot: Optional[Dict[str, Any]] = None
    cancel_requested: bool = False
    future: Optional[Future] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return JSON-safe public representation."""
        return {
            "job_id": self.job_id,
            "job_type": self.job_type,
            "status": self.status,
            "submitted_at": self.submitted_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "cancel_requested": self.cancel_requested,
            "error": self.error,
            "has_result": self.result is not None,
            "has_config": self.config_snapshot is not None,
        }


class JobManager:
    """Minimal async job manager for backend API."""

    def __init__(
        self,
        *,
        evaluation_runner: Callable[[EvaluationConfig], Any],
        matrix_runner: Callable[[EvaluationConfig, List[Dict[str, Any]]], Dict[str, Any]],
        max_workers: int = 2,
    ) -> None:
        self._evaluation_runner = evaluation_runner
        self._matrix_runner = matrix_runner
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="evaluator-webapi")
        self._jobs: Dict[str, JobRecord] = {}
        self._lock = Lock()

    def submit_evaluation(self, config: EvaluationConfig) -> JobRecord:
        return self._submit(
            "evaluation",
            lambda: self._evaluation_runner(config).to_dict(include_config=True),
            config_snapshot=config.to_dict(),
        )

    def submit_matrix(self, config: EvaluationConfig, test_setups: List[Dict[str, Any]]) -> JobRecord:
        return self._submit(
            "matrix",
            lambda: self._matrix_runner(config, test_setups),
            config_snapshot=config.to_dict(),
        )

    def _submit(
        self,
        job_type: str,
        runner: Callable[[], Dict[str, Any]],
        *,
        config_snapshot: Optional[Dict[str, Any]] = None,
    ) -> JobRecord:
        job_id = str(uuid4())
        job = JobRecord(job_id=job_id, job_type=job_type, config_snapshot=config_snapshot)
        with self._lock:
            self._jobs[job_id] = job
        job.future = self._executor.submit(self._run_job, job_id, runner)
        return job

    def _run_job(self, job_id: str, runner: Callable[[], Dict[str, Any]]) -> None:
        with self._lock:
            job = self._jobs[job_id]
            if job.cancel_requested:
                job.status = "cancelled"
                job.finished_at = _utc_now()
                return
            job.status = "running"
            job.started_at = _utc_now()

        try:
            result = runner()
        except Exception as exc:  # surfaced to API clients
            with self._lock:
                job = self._jobs[job_id]
                job.status = "failed"
                job.error = str(exc)
                job.finished_at = _utc_now()
            return

        with self._lock:
            job = self._jobs[job_id]
            if job.cancel_requested:
                job.status = "cancelled"
                job.finished_at = _utc_now()
                return
            job.result = result
            job.status = "completed"
            job.finished_at = _utc_now()

    def get(self, job_id: str) -> JobRecord:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            return self._jobs[job_id]

    def list_jobs(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [job.to_dict() for job in self._jobs.values()]

    def request_cancel(self, job_id: str) -> bool:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            job = self._jobs[job_id]
            job.cancel_requested = True
            if job.status == "queued" and job.future and job.future.cancel():
                job.status = "cancelled"
                job.finished_at = _utc_now()
                return True
            return job.status in {"cancelled", "completed", "failed"}


def _deep_merge_dict(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge patch dict into base dict."""
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge_dict(base[key], value)
        else:
            base[key] = value
    return base


def _nested_config(config: EvaluationConfig) -> Dict[str, Any]:
    """Return nested config shape for WebUI config creator."""
    return {
        "experiment_name": config.experiment_name,
        "output_dir": config.output_dir,
        "runtime": config.to_runtime_dict(),
        "experiment": config.to_experiment_dict(),
    }


def _create_config_options(provider_factory: Callable[[], ModelServiceProvider]) -> Dict[str, Any]:
    """Build form options for config creator UI."""
    provider = provider_factory()
    try:
        raw_models = provider.list_available_models()
    finally:
        provider.shutdown()

    def _normalize_model_entries(entries: Any) -> List[Dict[str, str]]:
        normalized: List[Dict[str, str]] = []
        if not isinstance(entries, list):
            return normalized
        for entry in entries:
            if isinstance(entry, dict):
                entry_type = str(entry.get("type", "")).strip()
                entry_name = str(entry.get("name", "")).strip() or entry_type
                if entry_type:
                    normalized.append({"type": entry_type, "name": entry_name})
                continue
            if isinstance(entry, str):
                value = entry.strip()
                if value:
                    normalized.append({"type": value, "name": value})
        return normalized

    normalized_models: Dict[str, List[Dict[str, str]]] = {}
    if isinstance(raw_models, dict):
        for family, entries in raw_models.items():
            normalized_models[str(family)] = _normalize_model_entries(entries)
    for required_family in ("asr", "text_embedding", "audio_embedding", "tts", "reranker"):
        normalized_models.setdefault(required_family, [])

    defaults = EvaluationConfig()
    return {
        "presets": list_presets(),
        "pipeline_modes": [mode.value for mode in PipelineMode],
        "dataset_types": [dataset_type.value for dataset_type in DatasetType],
        "dataset_sources": ["local", "huggingface", "custom"],
        "dataset_names": list_known_dataset_names(),
        "vector_db_types": [db.value for db in VectorDBType],
        "retrieval_modes": ["dense", "sparse", "hybrid"],
        "hybrid_fusion_methods": ["weighted", "rrf", "max_score"],
        "reranker_modes": ["none", "token_overlap", "cross_encoder"],
        "service_runtime": {
            "startup_mode": ["lazy", "eager"],
            "offload_policy": ["on_finish", "never"],
        },
        "tts_providers": sorted({entry["type"] for entry in normalized_models.get("tts", [])}),
        "models": normalized_models,
        "defaults": _nested_config(defaults),
    }


def _graph_preview(config: EvaluationConfig) -> Dict[str, Any]:
    profile = resolve_dataset_profile(config.data.dataset_name, config.data.dataset_type)
    graph = build_stage_graph(
        str(config.model.pipeline_mode),
        embedding_fusion_enabled=bool(config.embedding_fusion.enabled),
    )
    return {
        "mode": graph.mode,
        "nodes": [
            {"id": node.id, "stage": node.stage, "depends_on": list(node.depends_on)}
            for node in graph.nodes
        ],
        "levels": [[node.id for node in level] for level in graph.topological_levels()],
        "dataset_profile": {
            "name": profile.name,
            "dataset_type": str(profile.dataset_type),
            "requires_audio": profile.requires_audio,
            "requires_text": profile.requires_text,
            "supports_generation": profile.supports_generation,
            "evaluation_mode": profile.evaluation_mode,
            "recommended_pipeline_modes": list(profile.recommended_pipeline_modes),
            "pipeline_mode_supported": profile.supports_pipeline_mode(str(config.model.pipeline_mode)),
        },
    }


def _artifact_listing(output_dir: Optional[str], *, max_entries: int = 200) -> List[Dict[str, Any]]:
    if not output_dir:
        return []
    base = Path(output_dir)
    if not base.exists() or not base.is_dir():
        return []

    files: List[Dict[str, Any]] = []
    for path in sorted(base.rglob("*")):
        if not path.is_file():
            continue
        rel_path = str(path.relative_to(base))
        stat = path.stat()
        files.append(
            {
                "path": rel_path,
                "size_bytes": stat.st_size,
                "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
            }
        )
        if len(files) >= max_entries:
            break
    return files


def create_app(
    *,
    evaluation_runner: Callable[[EvaluationConfig], Any] = run_evaluation,
    matrix_runner: Callable[[EvaluationConfig, List[Dict[str, Any]]], Dict[str, Any]] = run_evaluation_matrix,
    provider_factory: Callable[[], ModelServiceProvider] = ModelServiceProvider,
) -> FastAPI:
    """Create FastAPI app for evaluator WebUI backend."""
    app = FastAPI(
        title="Evaluator Web API",
        version="0.2.0",
        description="HTTP backend for medical audio retrieval evaluation. "
                    "Supports async evaluation jobs, config management, "
                    "leaderboard queries, and live retrieval.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    jobs = JobManager(
        evaluation_runner=evaluation_runner,
        matrix_runner=matrix_runner,
    )

    @app.get("/", summary="Service info")
    def root() -> Dict[str, Any]:
        """Return service identity and useful navigation links."""
        return {
            "service": "evaluator-webapi",
            "status": "ok",
            "health": "/api/health",
            "docs": "/docs",
        }

    @app.get("/favicon.ico")
    def favicon() -> Response:
        return Response(status_code=204)

    @app.get("/api/health", response_model=HealthResponse, summary="Health check")
    def health() -> Dict[str, str]:
        """Liveness probe. Returns 200 if the service is running."""
        return {"status": "ok"}

    @app.get("/api/presets", summary="List presets")
    def presets() -> Dict[str, List[str]]:
        """Return available preset names for quick config creation."""
        return {"presets": list_presets()}

    @app.get("/api/models", summary="List available models")
    def list_models() -> Dict[str, Any]:
        """Return available models grouped by family (asr, text_embedding, etc.)."""
        provider = provider_factory()
        try:
            return provider.list_available_models()
        finally:
            provider.shutdown()

    @app.get("/api/models/{family}/{model_type}/params")
    def model_params(family: str, model_type: str) -> Dict[str, Any]:
        """Return parameter schema for a specific model type within a family.

        Response includes available sizes, default values, and extra params
        so the frontend can render a dynamic form.
        """
        from evaluator.models.registry import (
            asr_registry,
            text_embedding_registry,
            audio_embedding_registry,
            reranker_registry,
        )
        # Ensure models are registered
        from evaluator.models.asr import whisper, wav2vec2, faster_whisper  # noqa: F401
        from evaluator.models.t2e import labse, jina, clip, nemotron, bgem3  # noqa: F401
        from evaluator.models.a2e import attention_pool, clap_style, hubert, wavlm  # noqa: F401

        registry_map = {
            "asr": asr_registry,
            "text_embedding": text_embedding_registry,
            "audio_embedding": audio_embedding_registry,
            "reranker": reranker_registry,
        }
        registry = registry_map.get(family)
        if registry is None:
            raise HTTPException(status_code=404, detail=f"Unknown model family: {family}")
        if not registry.is_registered(model_type):
            raise HTTPException(status_code=404, detail=f"Unknown model type: {model_type}")

        return {
            "family": family,
            "model_type": model_type,
            "default_name": registry.get_default_name(model_type),
            "sizes": registry.get_sizes(model_type),
            "default_size": registry.get_default_size(model_type),
            "params_schema": registry.get_params_schema(model_type),
        }

    @app.get("/api/datasets")
    def list_datasets() -> Dict[str, Any]:
        """Return known dataset names, runtime specs, and type defaults."""
        from evaluator.datasets import list_known_dataset_names
        from evaluator.datasets.runtime import list_dataset_runtime_specs
        from evaluator.datasets.profiles import _DATASET_TYPE_DEFAULTS
        return {
            "known_datasets": list_known_dataset_names(),
            "runtime_specs": [
                {"id": s.id, "source": s.source, "description": s.description,
                 "required_fields": list(s.required_fields), "supports_corpus": s.supports_corpus}
                for s in list_dataset_runtime_specs()
            ],
            "dataset_type_defaults": {
                str(dt): {"evaluation_mode": p.evaluation_mode,
                           "recommended_pipeline_modes": list(p.recommended_pipeline_modes)}
                for dt, p in _DATASET_TYPE_DEFAULTS.items()
            },
        }

    @app.get("/api/services/status", summary="Service status with model inventory")
    def service_status() -> Dict[str, Any]:
        """Return available models and service health at check time."""
        provider = provider_factory()
        try:
            models = provider.list_available_models()
        finally:
            provider.shutdown()
        return {
            "status": "ok",
            "available_models": models,
            "checked_at": _utc_now(),
        }

    @app.get("/api/config/options", summary="Config form options")
    def config_options() -> Dict[str, Any]:
        """Return presets, pipeline modes, dataset types, model choices, and defaults for the config builder UI."""
        return _create_config_options(provider_factory)

    @app.post("/api/config/validate", summary="Validate config",
              responses={400: {"model": ErrorResponse}})
    def validate_config(payload: EvaluationJobRequest) -> Dict[str, Any]:
        """Validate and normalize an EvaluationConfig dict. Returns 400 on invalid config."""
        try:
            config = EvaluationConfig.from_dict(payload.config)
            if payload.auto_devices:
                config = config.with_auto_devices()
            check_backend_dependencies(config.vector_db)
            validate_dataset_runtime_config(
                config,
                retrieval_required=(str(config.model.pipeline_mode) != "asr_only"),
            )
            return {"config": config.to_dict()}
        except (ConfigurationError, ImportError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/graph/preview", summary="Preview pipeline DAG")
    def graph_preview(payload: EvaluationJobRequest) -> Dict[str, Any]:
        """Return stage graph nodes and levels for the given config."""
        try:
            config = EvaluationConfig.from_dict(payload.config)
            if payload.auto_devices:
                config = config.with_auto_devices()
            return _graph_preview(config)
        except ConfigurationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/config/create", summary="Create config from preset + patch",
              responses={400: {"model": ErrorResponse}})
    def create_config(payload: ConfigCreateRequest) -> Dict[str, Any]:
        """Create a new config by merging a patch dict over a preset base."""
        try:
            if payload.preset_name:
                base_config = EvaluationConfig.from_preset(payload.preset_name, validate=False)
            else:
                base_config = EvaluationConfig()

            merged = _deep_merge_dict(_nested_config(base_config), dict(payload.config_patch))
            config = EvaluationConfig.from_dict(merged)
            if payload.auto_devices:
                config = config.with_auto_devices()
            return {
                "config": _nested_config(config),
                "flat": config.to_dict(),
            }
        except ConfigurationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/jobs/evaluation", response_model=JobSubmitResponse,
              summary="Submit evaluation job",
              responses={400: {"model": ErrorResponse}})
    def submit_evaluation(payload: EvaluationJobRequest) -> Dict[str, str]:
        """Submit an async evaluation job. Poll /api/jobs/{job_id} for status."""
        try:
            config = EvaluationConfig.from_dict(payload.config)
            if payload.auto_devices:
                config = config.with_auto_devices()
            check_backend_dependencies(config.vector_db)
            validate_dataset_runtime_config(
                config,
                retrieval_required=(str(config.model.pipeline_mode) != "asr_only"),
            )
            job = jobs.submit_evaluation(config)
            return {"job_id": job.job_id}
        except (ConfigurationError, ImportError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/jobs/matrix", response_model=JobSubmitResponse,
              summary="Submit matrix evaluation job",
              responses={400: {"model": ErrorResponse}})
    def submit_matrix(payload: MatrixJobRequest) -> Dict[str, str]:
        """Submit an async matrix evaluation job with multiple test setups."""
        try:
            config = EvaluationConfig.from_dict(payload.base_config)
            if payload.auto_devices:
                config = config.with_auto_devices()
            check_backend_dependencies(config.vector_db)
            validate_dataset_runtime_config(
                config,
                retrieval_required=(str(config.model.pipeline_mode) != "asr_only"),
            )
            job = jobs.submit_matrix(config, payload.test_setups)
            return {"job_id": job.job_id}
        except (ConfigurationError, ImportError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/jobs", summary="List all jobs")
    def list_jobs() -> Dict[str, Any]:
        """Return all in-memory job records (queued, running, completed, failed)."""
        return {"jobs": jobs.list_jobs()}

    @app.get("/api/jobs/{job_id}", summary="Job status",
             responses={404: {"model": ErrorResponse}})
    def job_status(job_id: str) -> Dict[str, Any]:
        """Return current status of a job. Poll this endpoint until status is terminal."""
        try:
            job = jobs.get(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Job not found") from exc
        return job.to_dict()

    @app.post("/api/jobs/{job_id}/cancel", summary="Cancel job",
              responses={404: {"model": ErrorResponse}})
    def cancel_job(job_id: str) -> Dict[str, Any]:
        """Request cancellation of a running job."""
        try:
            accepted = jobs.request_cancel(job_id)
            status = jobs.get(job_id).status
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Job not found") from exc
        return {"job_id": job_id, "accepted": accepted, "status": status}

    @app.get("/api/jobs/{job_id}/result", summary="Job result",
             responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}})
    def job_result(job_id: str) -> Dict[str, Any]:
        """Return evaluation result for a completed job. 409 if still running."""
        try:
            job = jobs.get(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Job not found") from exc
        if job.status in {"queued", "running"}:
            raise HTTPException(status_code=409, detail="Job still running")
        if job.status == "failed":
            raise HTTPException(status_code=500, detail=job.error or "Job failed")
        if job.status == "cancelled":
            raise HTTPException(status_code=409, detail="Job cancelled")
        return {"job_id": job_id, "result": job.result}

    @app.get("/api/jobs/{job_id}/metadata", summary="Job metadata",
             responses={404: {"model": ErrorResponse}})
    def job_metadata(job_id: str) -> Dict[str, Any]:
        """Return config snapshot and metadata for a job."""
        try:
            job = jobs.get(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Job not found") from exc

        result_metadata = {}
        if isinstance(job.result, dict):
            result_metadata = job.result.get("metadata", {}) if isinstance(job.result.get("metadata"), dict) else {}
        return {
            "job": job.to_dict(),
            "config": job.config_snapshot,
            "metadata": result_metadata,
        }

    @app.get("/api/jobs/{job_id}/artifacts")
    def job_artifacts(job_id: str) -> Dict[str, Any]:
        try:
            job = jobs.get(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Job not found") from exc

        output_dir: Optional[str] = None
        if job.config_snapshot is not None:
            output_dir = job.config_snapshot.get("output_dir")
        artifacts = _artifact_listing(output_dir)
        return {"job_id": job_id, "output_dir": output_dir, "artifacts": artifacts}

    @app.get("/api/leaderboard", summary="Query leaderboard")
    def leaderboard(
        metric: str = "MRR",
        limit: int = 20,
        dataset_name: Optional[str] = None,
        pipeline_mode: Optional[str] = None,
        output_dir: str = "evaluation_results",
    ) -> Dict[str, Any]:
        store = ExperimentStore(db_path=str(Path(output_dir) / "leaderboard.sqlite"))
        rows = store.query_leaderboard(
            metric_name=metric,
            limit=limit,
            dataset_name=dataset_name,
            pipeline_mode=pipeline_mode,
        )
        return {
            "metric": metric,
            "rows": [
                {
                    "run_id": row.run_id,
                    "experiment_name": row.experiment_name,
                    "dataset_name": row.dataset_name,
                    "pipeline_mode": row.pipeline_mode,
                    "metric_value": row.metric_value,
                    "duration_seconds": row.duration_seconds,
                    "created_at": row.created_at,
                }
                for row in rows
            ],
        }

    @app.get("/api/leaderboard/runs", summary="List leaderboard runs")
    def leaderboard_runs(
        limit: int = 100,
        dataset_name: Optional[str] = None,
        pipeline_mode: Optional[str] = None,
        output_dir: str = "evaluation_results",
    ) -> Dict[str, Any]:
        store = ExperimentStore(db_path=str(Path(output_dir) / "leaderboard.sqlite"))
        runs = store.list_runs(
            limit=limit,
            dataset_name=dataset_name,
            pipeline_mode=pipeline_mode,
        )
        return {"runs": runs}

    @app.get("/api/leaderboard/runs/{run_id}", summary="Get run details",
             responses={404: {"model": ErrorResponse}})
    def leaderboard_run(
        run_id: int,
        output_dir: str = "evaluation_results",
    ) -> Dict[str, Any]:
        store = ExperimentStore(db_path=str(Path(output_dir) / "leaderboard.sqlite"))
        run = store.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return run

    @app.post("/api/live/query", summary="Ad-hoc retrieval query",
              responses={400: {"model": ErrorResponse}})
    def live_query(payload: LiveQueryRequest) -> Dict[str, Any]:
        """Run a single retrieval query against a configured pipeline."""
        if not payload.query_text.strip():
            raise HTTPException(status_code=400, detail="query_text must not be empty")
        if payload.k <= 0:
            raise HTTPException(status_code=400, detail="k must be positive")

        config = EvaluationConfig.from_dict(payload.config)
        if payload.auto_devices:
            config = config.with_auto_devices()

        cache_manager = CacheManager(
            cache_dir=config.cache.cache_dir,
            enabled=config.cache.enabled,
        )
        provider = provider_factory()
        try:
            bundle = create_pipeline_from_config(
                config,
                cache_manager,
                service_provider=provider,
            )
            if bundle.retrieval_pipeline is None or bundle.text_embedding_pipeline is None:
                raise HTTPException(
                    status_code=400,
                    detail="Live query requires retrieval and text embedding pipelines",
                )

            load_dataset(
                config,
                bundle.retrieval_pipeline,
                bundle.text_embedding_pipeline,
                cache_manager=cache_manager,
            )
            query_embedding = bundle.text_embedding_pipeline.process(payload.query_text)
            search_results = bundle.retrieval_pipeline.search_batch(
                np.array([query_embedding]),
                k=payload.k,
                query_texts=[payload.query_text],
            )[0]
            docs = []
            for row in search_results:
                docs.append(
                    {
                        "score": float(row.get("score", 0.0)),
                        "id": row.get("id"),
                        "title": row.get("title"),
                        "text": row.get("text", ""),
                    }
                )
            return {"query_text": payload.query_text, "k": payload.k, "results": docs}
        finally:
            provider.shutdown(offload=config.service_runtime.offload_policy != "never")

    return app
