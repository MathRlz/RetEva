"""Live query WebAPI endpoint."""

from typing import Any, Callable, Dict

import numpy as np
from fastapi import APIRouter, HTTPException

from evaluator import EvaluationConfig
from evaluator.pipeline import create_pipeline_from_config
from evaluator.services import ModelServiceProvider
from evaluator.services.evaluation_service import load_dataset
from evaluator.storage.cache import CacheManager
from evaluator.webapi.schemas import ErrorResponse, LiveQueryRequest


def build_live_router(
    provider_factory: Callable[[], ModelServiceProvider],
    *,
    pipeline_factory: Callable[..., Any] = create_pipeline_from_config,
    dataset_loader: Callable[..., Any] = load_dataset,
) -> APIRouter:
    router = APIRouter()

    @router.post(
        "/api/live/query",
        summary="Ad-hoc retrieval query",
        responses={400: {"model": ErrorResponse}},
    )
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
            bundle = pipeline_factory(
                config,
                cache_manager,
                service_provider=provider,
            )
            if bundle.retrieval_pipeline is None or bundle.text_embedding_pipeline is None:
                raise HTTPException(
                    status_code=400,
                    detail="Live query requires retrieval and text embedding pipelines",
                )

            dataset_loader(
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

    return router
