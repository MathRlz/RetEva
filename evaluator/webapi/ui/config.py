"""Config-page UI routes: the preset-driven config form + its validate / graph /
YAML / models / preset fragments. Mounted under ``/ui`` and ``/ui/config``.
"""

from __future__ import annotations

import html
from typing import Callable

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse
import yaml

from evaluator.services import ModelServiceProvider
from evaluator.webapi.form_builder import graph_preview
from evaluator.webapi.ui._common import _default_preset


from evaluator.webapi.form_builder import (
    _model_section,
    _prepared_config_or_error,
    _preset_form_context,
)


def _graph_diagram_context(preview: dict) -> dict:
    """Template vars for the read-only DAG preview partial (``_graph.html``): the preview
    payload as embedded JSON (the shared ``dag_view.js`` renders it via Drawflow in fixed
    mode, same architecture as the builder — BUILDER_UX.md B3) plus the levels for the
    no-JS / CDN-failure text-chips fallback."""
    import json as _json

    payload = _json.dumps(
        {
            "mode": preview.get("mode", ""),
            "levels": preview.get("levels", []),
            "nodes": [
                {
                    "id": n["id"],
                    "stage": n["stage"],
                    "bindings": n.get("bindings") or [],
                    "inputs": n.get("inputs") or [],
                    "outputs": n.get("outputs") or [],
                    "columns": n.get("columns") or [],
                    "inspect": n.get("inspect") or {},
                }
                for n in preview.get("nodes", [])
            ],
        }
    )
    return {
        # rendered with `| safe` inside a <script type="application/json"> block —
        # browsers do NOT decode HTML entities there, so Jinja's autoescape would
        # break JSON.parse; escape the one dangerous sequence instead.
        "preview_json": payload.replace("</", "<\\/"),
        "levels": preview.get("levels", []),
        "mode": preview.get("mode", ""),
    }


def register_config_routes(
    router: APIRouter,
    page,
    provider_factory: Callable[[], ModelServiceProvider],
) -> None:
    @router.get("/ui", include_in_schema=False)
    def ui_root() -> RedirectResponse:
        return RedirectResponse(url="/ui/config")

    @router.get("/ui/config", response_class=HTMLResponse, include_in_schema=False)
    def ui_config(request: Request) -> HTMLResponse:
        ctx = _preset_form_context(provider_factory, _default_preset())
        return page(request, "config.html", active="config", **ctx)

    @router.get("/ui/models", response_class=HTMLResponse, include_in_schema=False)
    def ui_models(
        request: Request, pipeline_mode: str = "asr_text_retrieval"
    ) -> HTMLResponse:
        # hx-include sends the whole form: the chosen model types ride along, so the
        # re-rendered section carries that model's registry-declared sizes + params
        # (same author-declared schema the builder uses).
        current = dict(request.query_params)
        return page(
            request,
            "_models.html",
            model_sections=_model_section(provider_factory, pipeline_mode, current),
        )

    @router.get("/ui/preset", response_class=HTMLResponse, include_in_schema=False)
    def ui_preset(request: Request, name: str = "") -> HTMLResponse:
        # hx-include sends the whole form as query params; preserve user fields
        # (dataset/paths) the chosen preset doesn't define.
        ctx = _preset_form_context(provider_factory, name, dict(request.query_params))
        return page(request, "_config_form.html", **ctx)

    @router.post("/ui/validate", response_class=HTMLResponse, include_in_schema=False)
    async def ui_validate(request: Request) -> HTMLResponse:
        config, error = _prepared_config_or_error(await request.form())
        if error is not None:
            return error
        return HTMLResponse('<p class="ok">Config valid ✓</p>')

    @router.post("/ui/graph", response_class=HTMLResponse, include_in_schema=False)
    async def ui_graph(request: Request) -> HTMLResponse:
        config, error = _prepared_config_or_error(await request.form())
        if error is not None:
            return error
        preview = graph_preview(config)
        return page(request, "_graph.html", **_graph_diagram_context(preview))

    @router.post("/ui/yaml", response_class=HTMLResponse, include_in_schema=False)
    async def ui_yaml(request: Request) -> HTMLResponse:
        """Render the full config the form produces as copy-able node-centric YAML."""
        config, error = _prepared_config_or_error(await request.form())
        if error is not None:
            return error
        from evaluator.config.graph_config import legacy_yaml_to_graph_yaml

        graph_dict = legacy_yaml_to_graph_yaml(config.to_dict(include_config=True))
        text = yaml.safe_dump(graph_dict, sort_keys=False, default_flow_style=False)
        return HTMLResponse(
            f'<section class="step"><h4>Config (YAML)</h4>'
            f'<pre class="yaml">{html.escape(text)}</pre></section>'
        )
