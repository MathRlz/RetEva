"""Minimalistic htmx UI for the evaluator.

Server-rendered Jinja2 templates + htmx (CDN) + Plotly (CDN, results only).
No build step, no node. UI routes reuse the same service/config helpers as the
JSON API and return HTML fragments for htmx swaps. Mounted at ``/ui``.

``build_ui_router`` builds the shared ``APIRouter`` + ``page`` renderer, then
delegates to the per-page registrar modules (config / builder / jobs / results /
tts), each of which owns the routes for its page-group.
"""

from __future__ import annotations

from typing import Callable

from fastapi import APIRouter
from fastapi.templating import Jinja2Templates

from evaluator.services import ModelServiceProvider
from evaluator.webapi.jobs import JobManager
from evaluator.webapi.ui._common import _TEMPLATES_DIR, make_page
from evaluator.webapi.ui.builder import register_builder_routes
from evaluator.webapi.ui.config import register_config_routes
from evaluator.webapi.ui.jobs import register_jobs_routes
from evaluator.webapi.ui.results import register_results_routes
from evaluator.webapi.ui.tts import register_tts_routes

__all__ = ["build_ui_router"]


def build_ui_router(
    provider_factory: Callable[[], ModelServiceProvider],
    jobs: JobManager,
) -> APIRouter:
    router = APIRouter()
    templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))
    page = make_page(templates)

    register_config_routes(router, page, provider_factory)
    register_builder_routes(router, page)
    register_jobs_routes(router, page, jobs)
    register_results_routes(router, page)
    register_tts_routes(router, page, provider_factory)

    return router
