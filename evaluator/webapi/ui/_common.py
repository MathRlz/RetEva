"""Shared helpers for the htmx UI route modules: the ``page`` renderer plus a couple of
request-independent helpers used across more than one page-group, so the per-page registrar
modules stay focused on route wiring."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from evaluator import EvaluationConfig
from evaluator.pipeline import build_graph_for_config, build_graph_from_spec
from evaluator.pipeline.graph import graph_label
from evaluator.pipeline.graph import (
    effective_inputs,
    effective_outputs,
    dataset_columns,
)

_TEMPLATES_DIR = Path(__file__).parent.parent / "templates"


def make_page(templates: Jinja2Templates):
    """Build the ``page(request, name, **ctx)`` template renderer used by all routes."""

    def page(request: Request, name: str, **ctx: Any) -> HTMLResponse:
        return templates.TemplateResponse(
            request, name, {"active": ctx.pop("active", ""), **ctx}
        )

    return page


def _graph_view_from_config(config_dict: dict):
    """Reconstruct the execution DAG (levels + per-node artifact I/O) from a stored run config,
    for the run-detail page. Tolerates config-snapshot format drift: ([], {}) on any error."""
    try:
        graph = None
        try:
            # Full reconstruction (incl. dataset column schema) when the stored config still
            # parses; falls back to wiring the stored nodes alone.
            graph = build_graph_for_config(
                EvaluationConfig.from_dict(dict(config_dict), validate=False)
            )
        except Exception:
            graph = None
        override = config_dict.get("graph_override") or {}
        if graph is None and override.get("nodes"):
            graph = build_graph_from_spec(
                override["nodes"],
                mode=graph_label(override) or "custom",
                edges=override.get("edges"),
            )
        if graph is None:
            return [], {}

        levels = [[n.id for n in level] for level in graph.topological_levels()]
        node_io = {}
        for n in graph.nodes:
            io = "{} → {}".format(
                ", ".join(effective_inputs(n.stage, n.params)) or "·",
                ", ".join(effective_outputs(n.stage, n.params)) or "·",
            )
            columns = dataset_columns(n.params)
            if columns:
                io += " · columns: " + ", ".join(
                    f"{c['name']}:{c['type']}({c['dtype']})" for c in columns
                )
            node_io[n.id] = io
        return levels, node_io
    except Exception:
        return [], {}
