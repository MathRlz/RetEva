"""Model-related WebAPI endpoints."""

from typing import Any, Callable, Dict

from fastapi import APIRouter, HTTPException

from evaluator.pipeline import ModelServiceProvider
from evaluator.webapi.utils import with_provider
from evaluator.models.registry import FAMILY_REGISTRIES
from evaluator.webapi.field_help import FIELD_HELP
from evaluator.models.llm.catalog import list_llm_models as _list


def build_models_router(provider_factory: Callable[[], ModelServiceProvider]) -> APIRouter:
    router = APIRouter()

    @router.get("/api/models", summary="List available models")
    def list_models() -> Dict[str, Any]:
        """Return available models grouped by family (asr, text_embedding, etc.)."""
        return with_provider(provider_factory, lambda p: p.list_available_models())

    @router.get("/api/models/{family}/{model_type}/params")
    def model_params(family: str, model_type: str) -> Dict[str, Any]:
        """Return parameter schema (sizes, defaults, extra params) for a model type, for the
        frontend's dynamic form."""

        registry = FAMILY_REGISTRIES.get(family)
        if registry is None:
            raise HTTPException(status_code=404, detail=f"Unknown model family: {family}")
        if not registry.is_registered(model_type):
            raise HTTPException(status_code=404, detail=f"Unknown model type: {model_type}")

        # Layer the shared glossary over the registry's per-field help: a model-author
        # DESCRIPTIONS entry wins (set in get_params_schema), else FIELD_HELP by field name,
        # so the builder shows a tooltip on model params just like the node-param switches.

        params_schema = registry.get_params_schema(model_type)
        for name, entry in params_schema.items():
            if "help" not in entry and name in FIELD_HELP:
                entry["help"] = FIELD_HELP[name]

        return {
            "family": family,
            "model_type": model_type,
            "default_name": registry.get_default_name(model_type),
            "sizes": registry.get_sizes(model_type),
            "default_size": registry.get_default_size(model_type),
            "params_schema": params_schema,
        }

    @router.get("/api/llm/models", summary="List LLM models for the builder picker")
    def list_llm_models(endpoint: str = "local", ollama_url: str | None = None) -> Dict[str, Any]:
        """Models for the global-LLM picker: ``endpoint=local`` probes Ollama live,
        ``endpoint=api`` returns a static OpenAI-compatible list; ``ollama_url`` overrides
        the probed server."""

        return _list(endpoint, ollama_url)

    return router
