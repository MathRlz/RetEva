"""Model-related WebAPI endpoints."""

from typing import Any, Callable, Dict

from fastapi import APIRouter, HTTPException

from evaluator.services import ModelServiceProvider
from evaluator.webapi.utils import with_provider


def build_models_router(provider_factory: Callable[[], ModelServiceProvider]) -> APIRouter:
    router = APIRouter()

    @router.get("/api/models", summary="List available models")
    def list_models() -> Dict[str, Any]:
        """Return available models grouped by family (asr, text_embedding, etc.)."""
        return with_provider(provider_factory, lambda p: p.list_available_models())

    @router.get("/api/models/{family}/{model_type}/params")
    def model_params(family: str, model_type: str) -> Dict[str, Any]:
        """Return parameter schema for a specific model type within a family.

        Response includes available sizes, default values, and extra params
        so the frontend can render a dynamic form.
        """
        import evaluator.models  # noqa: F401  (populates all registries)
        from evaluator.models.registry import FAMILY_REGISTRIES

        registry = FAMILY_REGISTRIES.get(family)
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

    @router.get("/api/pipeline/{mode}/required_models")
    def pipeline_required_models(mode: str) -> Dict[str, Any]:
        """Return required model fields for a pipeline mode (used by wizard step 3)."""
        from evaluator.pipeline.stage_graph import resolve_pipeline_mode_spec
        try:
            spec = resolve_pipeline_mode_spec(mode)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {
            "mode": spec.mode,
            "required_model_fields": list(spec.required_model_fields),
        }

    return router
