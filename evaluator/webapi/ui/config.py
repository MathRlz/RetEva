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
from evaluator.webapi.config_helpers import graph_preview
from evaluator.webapi.ui._common import _default_preset
from evaluator.webapi.ui_helpers import (
    _model_section,
    _prepared_config_or_error,
    _preset_form_context,
)


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
        return page(
            request,
            "_models.html",
            model_sections=_model_section(provider_factory, pipeline_mode, {}),
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
        node_io = {
            n["id"]: "{} → {}".format(
                ", ".join(n.get("inputs") or []) or "·",
                ", ".join(n.get("outputs") or []) or "·",
            )
            for n in preview.get("nodes", [])
        }
        return page(
            request,
            "_graph.html",
            levels=preview.get("levels", []),
            node_io=node_io,
        )

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
