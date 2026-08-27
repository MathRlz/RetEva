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
from evaluator.webapi.form_builder import graph_preview


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
                    # type/label override stage for aliased operators (e.g. "embed" → "audio_embedding")
                    # so the preview and seed use the concrete kind, not the raw operator.
                    "type": n.get("type") or n["stage"],
                    "label": n.get("label"),
                    "family": n.get("family"),
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


def _edit_nodes_json(name: str, preview: dict, config=None) -> str:
    """The meaningful nodes' editable form data (id/type/params/node_params/family/model_field) for
    the Config & Run inline parameter forms — embedded as JSON the page renders via NodeForm.
    Uses the expanded graph (graph_preview) so every node instance (e.g. multiple audio_embedding
    nodes from branches) gets its own form card. Structural plumbing is dropped."""
    import json as _json

    from evaluator.webapi.form_builder import default_model_for

    nodes = [
        {
            "id": n["id"], "type": n["type"], "label": n.get("label"),
            "params": n.get("params") or {},
            "node_params": n.get("node_params") or [],
            "family": n.get("family"), "model_field": n.get("model_field"),
            # bindings: [[artifact, source_id], …] — used by buildSpec() to reconstruct
            # explicit port edges so the run path wires correctly for multi-instance graphs.
            "bindings": n.get("bindings") or [],
            # input_ports: [{label, names, ...}, …] — buildSpec() needs this to map a bound
            # artifact name back to its canonical port label. A OneOf-collapsed port (e.g.
            # "query_vectors" accepting audio/text/fused query vectors) binds under the
            # PRODUCER's real artifact name, which differs from the port's own label; without
            # this, buildSpec() used the artifact name as the port key directly, which only
            # happened to work for plain 1:1 ports.
            "input_ports": n.get("input_ports") or [],
            # the form's empty model option names the inherited model — the LOADED config's
            # flat default here, not the dataclass default (audit: no bare "(default)")
            "default_model": default_model_for(n.get("model_field"), config),
        }
        for n in preview.get("nodes", []) if not n.get("structural")
    ]
    payload = _json.dumps({"mode": preview.get("mode"), "name": name, "nodes": nodes})
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
        except Exception as exc:  # noqa: BLE001 — surface a bad preset/build inline
            return HTMLResponse(
                f'<p class="error">Could not load preset: {html.escape(str(exc))}</p>'
            )
        errors, warnings = collect_problems(config)
        return page(
            request, "_config_run.html", preset_name=name,
            edit_nodes_json=_edit_nodes_json(name, preview, config),
            errors=errors, warnings=warnings, **_graph_diagram_context(preview)
        )

    @router.post("/ui/config-from-yaml", response_class=HTMLResponse, include_in_schema=False)
    async def ui_config_from_yaml(request: Request) -> HTMLResponse:
        """Upload a YAML config file and render its pipeline preview + editable param forms,
        same as /ui/config-preview but sourced from uploaded content rather than a preset name."""
        from evaluator import EvaluationConfig
        from evaluator.config.graph_config import build_evaluation_config_kwargs
        from evaluator.config.validation import collect_problems

        form = await request.form()
        raw_yaml = str(form.get("yaml") or "").strip()
        if not raw_yaml:
            return HTMLResponse('<p class="error">No YAML content received.</p>')
        try:
            import yaml as _yaml
            d = _yaml.safe_load(raw_yaml)
            if not isinstance(d, dict):
                raise ValueError("YAML must be a config mapping (key: value).")
            config = EvaluationConfig.from_dict(
                build_evaluation_config_kwargs(d), validate=False
            ).with_auto_devices()
            preview = graph_preview(config)
        except Exception as exc:  # noqa: BLE001
            return HTMLResponse(
                f'<p class="error">Could not parse config: {html.escape(str(exc))}</p>'
            )
        errors, warnings = collect_problems(config)
        return page(
            request, "_config_run.html", preset_name=None,
            edit_nodes_json=_edit_nodes_json("upload", preview, config),
            errors=errors, warnings=warnings, **_graph_diagram_context(preview),
        )

    @router.post("/ui/validate-graph", response_class=HTMLResponse, include_in_schema=False)
    async def ui_validate_graph(request: Request) -> HTMLResponse:
        """Re-validate the user's edited Config & Run graph (non-blocking): returns the problems
        fragment so they can fix the nodes and validate again before running. Runs the exact same
        pipeline ``/api/jobs/from-graph``/``/ui/run-graph`` run — ``build_validated_run_config``
        (structural completion, dataset choice, topology, embedding-space check) — mirroring the
        builder's ``/ui/validate-builder`` fix: "Valid ✓" here means the graph will actually build
        and run, not just that the edited fields alone are internally consistent, and now shows
        the applicable-metrics chips too (previously absent — this endpoint never analyzed the
        real graph at all)."""
        from evaluator.webapi.form_builder import graph_advice
        from evaluator.webapi.form_config import build_validated_run_config

        body = await request.json()
        spec = body.get("spec") or {}
        errors: list = []
        warnings: list = []
        metrics: list = []
        summary = None
        try:
            config, graph = build_validated_run_config(
                spec, experiment_name=body.get("name") or "webui", auto_devices=True
            )
            warnings, metrics = graph_advice(graph, config)
            summary = (f"Valid ✓ — {len(graph.topological_levels())} levels, "
                       f"{len(graph.nodes)} nodes")
            if not metrics:
                warnings.append(
                    "No metrics will be computed — the graph produces nothing a metric scores."
                )
        except Exception as exc:  # noqa: BLE001 — a broken graph is itself a problem to show
            errors = [str(exc)]
        return page(request, "_validation.html",
                    errors=errors, warnings=warnings, metrics=metrics, summary=summary)

    @router.post("/ui/validate-builder", response_class=HTMLResponse, include_in_schema=False)
    async def ui_validate_builder(request: Request) -> HTMLResponse:
        """Validate the builder canvas graph and render the SAME styled ``_validation.html`` the
        Config & Run flow uses (one validation module, no bespoke builder markup). Runs the exact
        same pipeline ``/api/jobs/from-graph``/``/ui/run-graph`` run — ``build_validated_run_config``
        (structural completion, dataset choice, topology, embedding-space check) — so "Valid ✓"
        here means the graph will actually build and run, not just that the drawn topology alone
        is internally consistent. That was a real, previously-silent gap: a canvas node renamed to
        an id a server-derived structural node also needs (e.g. "metrics") passed the OLD
        topology-only check but broke at Export/Run, since structural completion never ran here.
        Shares ``graph_advice`` (embedding warnings + applicable metrics + GT-missing) with the
        form-builder helpers."""
        from evaluator.webapi.form_builder import graph_advice
        from evaluator.webapi.form_config import build_validated_run_config

        spec = await request.json()  # the raw canvas spec {mode, nodes, edges, branches?}
        errors: list = []
        warnings: list = []
        metrics: list = []
        summary = None
        try:
            config, graph = build_validated_run_config(
                spec, experiment_name="validate", auto_devices=False
            )
            warnings, metrics = graph_advice(graph, config)
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
