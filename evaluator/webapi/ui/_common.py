"""Shared helpers for the htmx UI route modules.

The ``page`` renderer + a couple of request-independent helpers used across more
than one page-group live here so the per-page registrar modules stay focused on
route wiring.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from evaluator import list_presets

_TEMPLATES_DIR = Path(__file__).parent.parent / "templates"


def make_page(templates: Jinja2Templates):
    """Build the ``page(request, name, **ctx)`` template renderer used by all routes."""

    def page(request: Request, name: str, **ctx: Any) -> HTMLResponse:
        return templates.TemplateResponse(
            request, name, {"active": ctx.pop("active", ""), **ctx}
        )

    return page


def _default_preset() -> str:
    """First available config preset (the Config page loads prefilled from it)."""
    presets = list_presets()
    return presets[0] if presets else ""


def _graph_view_from_config(config_dict: dict):
    """Reconstruct the execution DAG (levels + per-node artifact I/O) from a stored
    run config, for the run-detail page. Only needs pipeline_mode (+ fusion / explicit
    override), so it tolerates config-snapshot format drift. ([], {}) on any error."""
    try:
        from evaluator.pipeline import (
            build_graph_from_spec,
            build_stage_graph,
            get_stage_node_def,
        )

        model = config_dict.get("model") or {}
        mode = config_dict.get("pipeline_mode") or model.get("pipeline_mode")
        if not mode:
            return [], {}
        mode = str(mode)
        override = config_dict.get("graph_override")
        if override and override.get("nodes"):
            graph = build_graph_from_spec(
                override["nodes"], mode=mode, edges=override.get("edges")
            )
        else:
            fusion = bool(
                (config_dict.get("embedding_fusion") or {}).get("enabled")
                or (config_dict.get("features") or {})
                .get("embedding_fusion", {})
                .get("enabled")
            )
            graph = build_stage_graph(mode, embedding_fusion_enabled=fusion)
        levels = [[n.id for n in level] for level in graph.topological_levels()]
        node_io = {
            n.id: "{} → {}".format(
                ", ".join(get_stage_node_def(n.stage).inputs) or "·",
                ", ".join(get_stage_node_def(n.stage).outputs) or "·",
            )
            for n in graph.nodes
        }
        return levels, node_io
    except Exception:
        return [], {}
