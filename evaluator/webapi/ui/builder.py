"""Visual pipeline builder page (E1). Mounted at ``/ui/builder``."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse


def register_builder_routes(router: APIRouter, page) -> None:
    @router.get("/ui/builder", response_class=HTMLResponse, include_in_schema=False)
    def ui_builder(request: Request) -> HTMLResponse:
        """Visual pipeline builder (E1): Drawflow canvas + palette (from /api/graph/nodes)."""
        return page(request, "builder.html", active="builder")
