"""Dataset-related WebAPI endpoints."""

from typing import Any, Dict

from fastapi import APIRouter, HTTPException

# DatasetType value → coarse modality for grouping the picker (Audio / Text / Multimodal).
_MODALITY_BY_TYPE = {
    "audio_transcription": "audio",
    "audio_query_retrieval": "audio",
    "text_query_retrieval": "text",
    "question_answering": "text",
    "passage_ranking": "text",
    "multimodal_qa": "multimodal",
}


def _modality(dataset_type: str) -> str:
    return _MODALITY_BY_TYPE.get(str(dataset_type), "other")


def build_datasets_router() -> APIRouter:
    router = APIRouter()

    @router.get("/api/dataset/{dataset_id}/fields")
    def dataset_fields_endpoint(dataset_id: str) -> Dict[str, Any]:
        """Return DatasetDescriptor metadata for a registered dataset id (used by wizard step 2)."""
        from evaluator.datasets.descriptor import get_descriptor, list_registered_datasets
        desc = get_descriptor(dataset_id)
        if desc is None:
            known = ", ".join(list_registered_datasets())
            raise HTTPException(status_code=404, detail=f"Unknown dataset '{dataset_id}'. Known: {known}")
        from evaluator.config.graph_config import _DATA_FIELD_TO_KEY
        from evaluator.pipeline.artifacts import artifact_modality, is_registered

        return {
            "id": desc.id,
            "description": desc.description,
            "requires_audio": desc.requires_audio,
            "requires_text": desc.requires_text,
            "supports_generation": desc.supports_generation,
            "evaluation_mode": desc.evaluation_mode,
            "compatible_pipeline_modes": list(desc.compatible_pipeline_modes),
            "required_data_fields": list(desc.required_data_fields),
            "splits": list(desc.splits),
            "default_split": desc.default_split,
            "default_metrics": list(desc.default_metrics),
            # Column schema (§2): what the dataset node shows on the DAG.
            "fields": [
                {
                    "name": name,
                    "artifact": artifact,
                    "type": (
                        str(artifact_modality(artifact).value)
                        if is_registered(artifact)
                        else "?"
                    ),
                }
                for name, artifact in (desc.fields or {}).items()
            ],
            # Required settings as the builder's node-param keys (the node-centric
            # YAML vocabulary) — the UI hardcodes nothing.
            "required_settings": [
                {"key": _DATA_FIELD_TO_KEY.get(field, field), "field": field}
                for field in desc.required_data_fields
            ],
        }

    @router.get("/api/datasets")
    def list_datasets() -> Dict[str, Any]:
        """Return known datasets with grouping metadata (modality + domain), plus the bare
        id list (back-compat) and type defaults."""
        from evaluator.datasets import list_known_dataset_names
        from evaluator.datasets.descriptor import get_descriptor
        from evaluator.datasets.profiles import _DATASET_TYPE_DEFAULTS

        ids = list_known_dataset_names()
        datasets = []
        for did in ids:
            d = get_descriptor(did)
            if d is None:
                datasets.append({"id": did, "modality": "other", "domain": "general"})
                continue
            datasets.append({
                "id": did,
                "dataset_type": str(d.dataset_type),
                "modality": _modality(d.dataset_type),
                "domain": getattr(d, "domain", "general"),
                "description": d.description,
                "requires_audio": d.requires_audio,
                "requires_text": d.requires_text,
                "supports_generation": d.supports_generation,
                "compatible_pipeline_modes": list(d.compatible_pipeline_modes),
            })
        return {
            "known_datasets": ids,  # back-compat (bare ids)
            "datasets": datasets,   # rich entries for the grouped picker
            "dataset_type_defaults": {
                str(dt): {
                    "evaluation_mode": p.evaluation_mode,
                    "recommended_pipeline_modes": list(p.recommended_pipeline_modes),
                }
                for dt, p in _DATASET_TYPE_DEFAULTS.items()
            },
        }

    return router
