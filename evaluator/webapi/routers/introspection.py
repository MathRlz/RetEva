"""Introspection schema endpoint — the single registry/enum-sourced contract the web UI builds
itself from.

Every list here comes from a core registry or a `config.types` enum/tuple, so a model / metric /
strategy / dataset / node / vector-store registered in the evaluator appears in the UI with **no UI
edit** (no drift). The front-end fetches `/api/introspection/schema` once and renders its option
lists, palettes, and forms against it.
"""

from typing import Any, Dict, List

from fastapi import APIRouter


def _model_families() -> Dict[str, List[Dict[str, Any]]]:
    """Per-family registered model types with sizes + metadata (registry-driven)."""
    from evaluator.models.registry import (
        asr_registry,
        text_embedding_registry,
        audio_embedding_registry,
        reranker_registry,
        tts_registry,
    )

    registries = {
        "asr": asr_registry,
        "text_embedding": text_embedding_registry,
        "audio_embedding": audio_embedding_registry,
        "reranker": reranker_registry,
        "tts": tts_registry,
    }
    out: Dict[str, List[Dict[str, Any]]] = {}
    for family, reg in registries.items():
        types: List[Dict[str, Any]] = []
        for mtype in reg.list_types():
            meta = reg.get_metadata(mtype) or {}
            types.append({
                "type": mtype,
                "sizes": reg.get_sizes(mtype) or {},
                "description": meta.get("description"),
                "default_name": meta.get("default_name"),
            })
        out[family] = types
    return out


def _datasets() -> List[Dict[str, Any]]:
    """Registered datasets with their column schema + coarse modality (descriptor-driven)."""
    from evaluator.config.types import DATASET_TYPE_MODALITY
    from evaluator.datasets import list_known_dataset_names
    from evaluator.datasets.descriptor import get_descriptor
    from evaluator.pipeline.artifacts import artifact_modality, is_registered

    out: List[Dict[str, Any]] = []
    for did in list_known_dataset_names():
        d = get_descriptor(did)
        if d is None:
            continue
        out.append({
            "id": did,
            "dataset_type": str(d.dataset_type),
            "modality": DATASET_TYPE_MODALITY.get(str(d.dataset_type), "other"),
            "domain": getattr(d, "domain", "general"),
            "description": d.description,
            "fields": [
                {
                    "name": name,
                    "artifact": artifact,
                    "type": (
                        str(artifact_modality(artifact).value)
                        if is_registered(artifact) else "?"
                    ),
                }
                for name, artifact in (d.fields or {}).items()
            ],
            "required_data_fields": list(d.required_data_fields),
            "default_metrics": list(d.default_metrics),
            "splits": list(d.splits),
        })
    return out


def build_introspection_router() -> APIRouter:
    router = APIRouter()

    @router.get(
        "/api/introspection/schema",
        summary="The full registry/enum-sourced UI schema (single source of truth)",
    )
    def introspection_schema() -> Dict[str, Any]:
        from evaluator.config.types import (
            RETRIEVAL_MODES,
            RERANKER_MODES,
            SERVICE_STARTUP_MODES,
            SERVICE_OFFLOAD_POLICIES,
            DATASET_SOURCES,
        )
        from evaluator.evaluation.metric_registry import list_metrics
        from evaluator.evaluation.query_correction import list_correctors
        from evaluator.evaluation.results import HEADLINE_METRICS
        from evaluator.models.retrieval import list_fusions
        from evaluator.models.retrieval.query.optimization import list_combine_strategies
        from evaluator.models.retrieval.rag.strategies import DistanceMetric
        from evaluator.pipeline.artifacts import list_artifacts
        from evaluator.pipeline.graph.templates import list_graph_templates
        from evaluator.storage.registry import list_vector_stores
        from evaluator.webapi.field_help import FIELD_HELP
        from evaluator.webapi.form_builder import node_catalogue

        return {
            "graph_templates": list_graph_templates(),
            "model_families": _model_families(),
            "datasets": _datasets(),
            "metrics": [
                {
                    "name": m.name,
                    "scored": m.scored,
                    "gt": m.gt,
                    "inputs": list(m.inputs),
                    "higher_is_better": m.higher_is_better,
                }
                for m in list_metrics()
            ],
            "headline_metrics": list(HEADLINE_METRICS),
            "artifacts": [
                {
                    "name": a.name,
                    "modality": str(a.modality.value),
                    "is_source": a.is_source,
                }
                for a in list_artifacts()
            ],
            "vector_stores": list_vector_stores(),
            "fusion_methods": list_fusions(),
            "combine_strategies": list_combine_strategies(),
            "correctors": list_correctors(),
            "retrieval_modes": list(RETRIEVAL_MODES),
            "reranker_modes": list(RERANKER_MODES),
            "dataset_sources": list(DATASET_SOURCES),
            "startup_modes": list(SERVICE_STARTUP_MODES),
            "offload_policies": list(SERVICE_OFFLOAD_POLICIES),
            "distance_metrics": [d.value for d in DistanceMetric],
            "field_help": dict(FIELD_HELP),
            "node_catalogue": node_catalogue(),
        }

    return router
