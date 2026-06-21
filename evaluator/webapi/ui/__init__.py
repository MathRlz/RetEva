"""Minimalistic htmx UI for the evaluator.

Server-rendered Jinja2 templates + htmx (CDN) + Plotly (CDN, results only).
No build step, no node. UI routes reuse the same service/config helpers as the
JSON API and return HTML fragments for htmx swaps. Mounted at ``/ui``.

``build_ui_router`` builds the shared ``APIRouter`` + ``page`` renderer, then
delegates to the per-page registrar modules (config / builder / jobs /
results), each of which owns the routes for its page-group.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable

from fastapi import APIRouter
from fastapi.templating import Jinja2Templates

# Static dir (sibling of this `ui` package) for content-derived cache-bust tokens.
_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


def _asset_version(filename: str) -> str:
    """A short token that changes when ``static/<filename>`` changes (mtime+size), so a JS/CSS
    edit auto-invalidates the browser cache — no hand-bumped ``?v=`` strings to forget."""
    try:
        st = (_STATIC_DIR / filename).stat()
        return hashlib.md5(f"{st.st_mtime_ns}:{st.st_size}".encode()).hexdigest()[:10]
    except OSError:
        return "0"


from evaluator.services import ModelServiceProvider
from evaluator.webapi.jobs import JobManager
from evaluator.webapi.ui._common import _TEMPLATES_DIR, make_page
from evaluator.webapi.ui.builder import register_builder_routes
from evaluator.webapi.ui.config import register_config_routes
from evaluator.webapi.ui.jobs import register_jobs_routes
from evaluator.webapi.ui.results import register_results_routes

__all__ = ["build_ui_router"]


def build_ui_router(
    provider_factory: Callable[[], ModelServiceProvider],
    jobs: JobManager,
) -> APIRouter:
    router = APIRouter()
    from evaluator.evaluation.results import HEADLINE_METRICS

    templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))
    templates.env.globals["asset_version"] = _asset_version
    # Canonical headline-metric order (single source — see evaluation/results.py), so the
    # status chips never drift from the console summary / report priority.
    templates.env.globals["headline_metrics"] = list(HEADLINE_METRICS)
    page = make_page(templates)

    register_config_routes(router, page, provider_factory)
    register_builder_routes(router, page)
    register_jobs_routes(router, page, jobs)
    register_results_routes(router, page)

    return router
