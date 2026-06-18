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
        )

        model = config_dict.get("model") or {}
        # The template is on graph_override now; tolerate legacy snapshots' flat pipeline_mode.
        mode = (
            (config_dict.get("graph_override") or {}).get("template")
            or config_dict.get("pipeline_mode")
            or model.get("pipeline_mode")
        )
        if not mode:
            return [], {}
        mode = str(mode)
        graph = None
        try:
            # Full reconstruction (incl. dataset column schema) when the stored
            # config still parses; falls back to the tolerant mode-only path.
            from evaluator import EvaluationConfig
            from evaluator.pipeline import build_graph_for_config

            graph = build_graph_for_config(
                EvaluationConfig.from_dict(dict(config_dict), validate=False)
            )
        except Exception:
            graph = None
        override = config_dict.get("graph_override")
        if graph is None and override and override.get("nodes"):
            graph = build_graph_from_spec(
                override["nodes"], mode=mode, edges=override.get("edges")
            )
        elif graph is None:
            fusion = bool(
                (config_dict.get("embedding_fusion") or {}).get("enabled")
                or (config_dict.get("features") or {})
                .get("embedding_fusion", {})
                .get("enabled")
            )
            graph = build_stage_graph(mode, embedding_fusion_enabled=fusion)
        from evaluator.pipeline.stage_graph import (
            _effective_inputs,
            _effective_outputs,
            dataset_columns,
        )

        levels = [[n.id for n in level] for level in graph.topological_levels()]
        node_io = {}
        for n in graph.nodes:
            io = "{} → {}".format(
                ", ".join(_effective_inputs(n.stage, n.params)) or "·",
                ", ".join(_effective_outputs(n.stage, n.params)) or "·",
            )
            columns = dataset_columns(n.params)
            if columns:
                io += " · columns: " + ", ".join(
                    f"{c['name']}:{c['type']}" for c in columns
                )
            node_io[n.id] = io
        return levels, node_io
    except Exception:
        return [], {}
