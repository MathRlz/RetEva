"""Config-related WebAPI endpoints."""

from typing import Any, Callable, Dict

from fastapi import APIRouter, HTTPException

from evaluator import ConfigurationError
from evaluator.pipeline.stage_graph import PIPELINE_MODE_SPECS, resolve_pipeline_mode_spec
from evaluator.datasets.descriptor import list_registered_datasets, get_descriptor
from evaluator.services import ModelServiceProvider
from evaluator.webapi.config_helpers import (
    create_config_options,
    deep_merge_dict,
    graph_preview,
    load_config,
    nested_config,
    prepare_run_config,
)
from evaluator.webapi.schemas import ConfigCreateRequest, ErrorResponse, EvaluationJobRequest


def build_config_router(provider_factory: Callable[[], ModelServiceProvider]) -> APIRouter:
    router = APIRouter()

    @router.get("/api/presets", summary="List presets")
    def presets() -> Dict[str, list[str]]:
        """Return available preset names for quick config creation."""
        from evaluator import list_presets
        return {"presets": list_presets()}

    @router.get("/api/config/options", summary="Config form options")
    def config_options() -> Dict[str, Any]:
        """Return presets, pipeline modes, dataset types, model choices, and defaults for the config builder UI."""
        return create_config_options(provider_factory)

    @router.get("/api/config/schema", summary="Config schema for wizard UI")
    def config_schema() -> Dict[str, Any]:
        """Return structured schema: pipeline modes → required model fields + compatible datasets,
        datasets → field requirements + default metrics. Used by frontend wizard."""
        modes: Dict[str, Any] = {}
        for mode in sorted(PIPELINE_MODE_SPECS.keys()):
            try:
                spec = resolve_pipeline_mode_spec(mode)
            except ValueError:
                continue
            compatible_datasets = [
                ds_id for ds_id in list_registered_datasets()
                if (desc := get_descriptor(ds_id)) and desc.supports_pipeline_mode(mode)
            ]
            modes[mode] = {
                "required_model_fields": list(spec.required_model_fields),
                "compatible_datasets": compatible_datasets,
            }

        datasets: Dict[str, Any] = {}
        for ds_id in list_registered_datasets():
            desc = get_descriptor(ds_id)
            if desc is None:
                continue
            datasets[ds_id] = {
                "description": desc.description,
                "requires_audio": desc.requires_audio,
                "requires_text": desc.requires_text,
                "supports_generation": desc.supports_generation,
                "evaluation_mode": desc.evaluation_mode,
                "compatible_pipeline_modes": list(desc.compatible_pipeline_modes),
                "required_data_fields": list(desc.required_data_fields),
                "default_metrics": list(desc.default_metrics),
            }

        return {"pipeline_modes": modes, "datasets": datasets}

    @router.post(
        "/api/config/validate",
        summary="Validate config",
        responses={400: {"model": ErrorResponse}},
    )
    def validate_config(payload: EvaluationJobRequest) -> Dict[str, Any]:
        """Validate and normalize an EvaluationConfig dict. Returns 400 on invalid config."""
        try:
            config = prepare_run_config(payload.config, auto_devices=payload.auto_devices)
            return {"config": config.to_dict()}
        except (ConfigurationError, ImportError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/api/graph/preview", summary="Preview pipeline DAG")
    def graph_preview_endpoint(payload: EvaluationJobRequest) -> Dict[str, Any]:
        """Return stage graph nodes and levels for the given config."""
        try:
            config = load_config(payload.config, auto_devices=payload.auto_devices)
            return graph_preview(config)
        except ConfigurationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post(
        "/api/config/create",
        summary="Create config from preset + patch",
        responses={400: {"model": ErrorResponse}},
    )
    def create_config(payload: ConfigCreateRequest) -> Dict[str, Any]:
        """Create a new config by merging a patch dict over a preset base."""
        try:
            from evaluator import EvaluationConfig
            if payload.preset_name:
                base_config = EvaluationConfig.from_preset(payload.preset_name, validate=False)
            else:
                base_config = EvaluationConfig()

            merged = deep_merge_dict(nested_config(base_config), dict(payload.config_patch))
            config = EvaluationConfig.from_dict(merged)
            if payload.auto_devices:
                config = config.with_auto_devices()
            return {
                "config": nested_config(config),
                "flat": config.to_dict(),
            }
        except ConfigurationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return router
