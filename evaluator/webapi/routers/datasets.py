"""Dataset-related WebAPI endpoints."""

from typing import Any, Dict

from fastapi import APIRouter, HTTPException


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
        return {
            "id": desc.id,
            "description": desc.description,
            "requires_audio": desc.requires_audio,
            "requires_text": desc.requires_text,
            "supports_generation": desc.supports_generation,
            "evaluation_mode": desc.evaluation_mode,
            "compatible_pipeline_modes": list(desc.compatible_pipeline_modes),
            "required_data_fields": list(desc.required_data_fields),
            "default_metrics": list(desc.default_metrics),
        }

    @router.get("/api/datasets")
    def list_datasets() -> Dict[str, Any]:
        """Return known dataset names, runtime specs, and type defaults."""
        from evaluator.datasets import list_known_dataset_names
        from evaluator.datasets.runtime import list_dataset_runtime_specs
        from evaluator.datasets.profiles import _DATASET_TYPE_DEFAULTS
        return {
            "known_datasets": list_known_dataset_names(),
            "runtime_specs": [
                {
                    "id": s.id,
                    "source": s.source,
                    "description": s.description,
                    "required_fields": list(s.required_fields),
                    "supports_corpus": s.supports_corpus,
                }
                for s in list_dataset_runtime_specs()
            ],
            "dataset_type_defaults": {
                str(dt): {
                    "evaluation_mode": p.evaluation_mode,
                    "recommended_pipeline_modes": list(p.recommended_pipeline_modes),
                }
                for dt, p in _DATASET_TYPE_DEFAULTS.items()
            },
        }

    return router
