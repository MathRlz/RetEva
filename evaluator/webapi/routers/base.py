"""Base routes: root, health, service status."""

from typing import Any, Callable, Dict

from fastapi import APIRouter, Response

from evaluator.services import ModelServiceProvider
from evaluator.webapi.schemas import HealthResponse


def build_base_router(provider_factory: Callable[[], ModelServiceProvider]) -> APIRouter:
    router = APIRouter()

    @router.get("/", summary="Service info")
    def root() -> Dict[str, Any]:
        """Return service identity and useful navigation links."""
        return {
            "service": "evaluator-webapi",
            "status": "ok",
            "health": "/api/health",
            "docs": "/docs",
        }

    @router.get("/favicon.ico")
    def favicon() -> Response:
        return Response(status_code=204)

    @router.get("/api/health", response_model=HealthResponse, summary="Health check")
    def health() -> Dict[str, str]:
        """Liveness probe. Returns 200 if the service is running."""
        return {"status": "ok"}

    return router
