"""Config-page UI routes: the simplified Config & Run flow (pick a preset → preview its
execution DAG → run / open in builder). Mounted under ``/ui`` and ``/ui/config``.
"""

from __future__ import annotations

import html
from typing import Callable

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from evaluator import list_presets
from evaluator.services import ModelServiceProvider
from evaluator.webapi.form_builder import config_to_canvas_spec, graph_preview


def _graph_diagram_context(preview: dict) -> dict:
    """Template vars for the read-only DAG preview partial (``_graph.html``): the preview
    payload as embedded JSON (the shared ``dag_view.js`` renders it via Drawflow in fixed
    mode, same architecture as the builder) plus the levels for the
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
                    # the default-simplified preview hides structural plumbing (metrics/report/
                    # finalize/sinks) — _graph.html filters on this flag, so it MUST ride along.
                    "structural": n.get("structural", False),
                    "bindings": n.get("bindings") or [],
                    "inputs": n.get("inputs") or [],
                    # input_ports collapse a OneOf chain to ONE port; optional_inputs carry the
                    # GT side-channels. Dropping either broke the preview: the chain exploded
                    # into N dangling ports and every optional-GT edge silently vanished.
                    "input_ports": n.get("input_ports") or [],
                    "optional_inputs": n.get("optional_inputs") or [],
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
        "mode": preview.get("mode", ""),
    }


def _edit_nodes_json(name: str, canvas: dict, config=None) -> str:
    """The meaningful nodes' editable form data (id/type/params/node_params/family/model_field) for
    the Config & Run inline parameter forms — embedded as JSON the page renders via NodeForm. The
    structural plumbing is dropped (auto-derived); the run re-derives it (graph_spec_to_config)."""
    import json as _json

    from evaluator.webapi.form_builder import default_model_for

    nodes = [
        {
            "id": n["id"], "type": n["type"], "label": n.get("label"),
            "params": n.get("params") or {},
            "node_params": n.get("node_params") or [],
            "family": n.get("family"), "model_field": n.get("model_field"),
            # the form's empty model option names the inherited model — the LOADED config's
            # flat default here, not the dataclass default (audit: no bare "(default)")
            "default_model": default_model_for(n.get("model_field"), config),
        }
        for n in canvas.get("nodes", []) if not n.get("structural")
    ]
    payload = _json.dumps({"mode": canvas.get("mode"), "name": name, "nodes": nodes})
    return payload.replace("</", "<\\/")  # safe inside <script type="application/json">


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
        return page(
            request, "config.html", active="config",
            options={"presets": list_presets()},
        )

    @router.post("/ui/config-preview", response_class=HTMLResponse, include_in_schema=False)
    async def ui_config_preview(request: Request) -> HTMLResponse:
        """Simplified Config & Run: a chosen preset → its execution DAG preview (default-simplified,
        with 'View full DAG' + 'Open in builder') + a Run button. Reuses config→DAG (graph_preview)
        and DAG→simplified (the structural flag) — no separate simplified-from-config path."""
        from evaluator import EvaluationConfig, get_preset
        from evaluator.config.validation import collect_problems

        name = (await request.form()).get("name") or ""
        if not name:
            return HTMLResponse(
                '<p class="muted">Pick a configuration to preview its pipeline.</p>'
            )
        try:
            # validate=False: build the preview + forms even if the config has problems (e.g. a
            # missing dataset path) — the problems show below so the user can fix + revalidate.
            # auto-devices so the validation matches the run path (no spurious cuda warnings).
            config = (EvaluationConfig.from_dict(get_preset(name), validate=False)
                      .with_auto_devices())
            preview = graph_preview(config)
            canvas = config_to_canvas_spec(config)
        except Exception as exc:  # noqa: BLE001 — surface a bad preset/build inline
            return HTMLResponse(
                f'<p class="error">Could not load preset: {html.escape(str(exc))}</p>'
            )
        errors, warnings = collect_problems(config)
        return page(
            request, "_config_run.html", preset_name=name,
            edit_nodes_json=_edit_nodes_json(name, canvas, config),
            errors=errors, warnings=warnings, **_graph_diagram_context(preview)
        )

    @router.post("/ui/validate-graph", response_class=HTMLResponse, include_in_schema=False)
    async def ui_validate_graph(request: Request) -> HTMLResponse:
        """Re-validate the user's edited Config & Run graph (non-blocking): returns the problems
        fragment so they can fix the nodes and validate again before running."""
        from evaluator import EvaluationConfig
        from evaluator.config.validation import collect_problems
        from evaluator.webapi.form_config import graph_spec_to_config_dict

        body = await request.json()
        spec = body.get("spec") or {}
        try:
            config_dict = graph_spec_to_config_dict(
                spec, experiment_name=body.get("name") or "webui")
            config = EvaluationConfig.from_dict(config_dict, validate=False).with_auto_devices()
            errors, warnings = collect_problems(config)
        except Exception as exc:  # noqa: BLE001 — a broken graph is itself a problem to show
            errors, warnings = [str(exc)], []
        return page(request, "_validation.html", errors=errors, warnings=warnings)

    @router.post("/ui/validate-builder", response_class=HTMLResponse, include_in_schema=False)
    async def ui_validate_builder(request: Request) -> HTMLResponse:
        """Validate the builder canvas graph and render the SAME styled ``_validation.html`` the
        Config & Run flow uses (one validation module, no bespoke builder markup). Builds the
        topology directly (no run config / dataset needed, so a graph-in-progress validates) and
        shares ``build_canvas_graph`` + ``graph_advice`` (embedding warnings + applicable metrics
        + GT-missing) with the form-builder helpers."""
        from evaluator import EvaluationConfig
        from evaluator.webapi.form_builder import build_canvas_graph, graph_advice

        spec = await request.json()  # the raw canvas spec {mode, nodes, edges, branches?}
        errors: list = []
        warnings: list = []
        metrics: list = []
        summary = None
        try:
            graph = build_canvas_graph(spec)
            warnings, metrics = graph_advice(graph, EvaluationConfig())
            summary = (f"Valid ✓ — {len(graph.topological_levels())} levels, "
                       f"{len(graph.nodes)} nodes")
            if not metrics:
                warnings.append(
                    "No metrics will be computed — the graph produces nothing a metric scores."
                )
        except Exception as exc:  # noqa: BLE001 — a broken graph is the problem to show
            errors = [str(exc)]
        return page(request, "_validation.html",
                    errors=errors, warnings=warnings, metrics=metrics, summary=summary)
