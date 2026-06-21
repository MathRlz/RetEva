"""Config-related WebAPI endpoints."""

from typing import Any, Callable, Dict

from fastapi import APIRouter, HTTPException

from evaluator.pipeline.graph.templates import GRAPH_TEMPLATES, graph_template
from evaluator.datasets.descriptor import list_registered_datasets, get_descriptor
from evaluator.services import ModelServiceProvider
from evaluator.webapi.form_builder import (
    config_to_canvas_spec,
    create_config_options,
    deep_merge_dict,
    graph_preview,
    graph_render_payload,
    load_config,
    nested_config,
    prepare_run_config,
    render_node,
    required_model_fields_for,
)
from evaluator.webapi.schemas import (
    ConfigCreateRequest,
    ErrorResponse,
    EvaluationJobRequest,
)


def build_config_router(
    provider_factory: Callable[[], ModelServiceProvider],
) -> APIRouter:
    router = APIRouter()

    @router.get("/api/presets", summary="List presets")
    def presets() -> Dict[str, list[str]]:
        """Return available preset names for quick config creation."""
        from evaluator import list_presets

        return {"presets": list_presets()}

    @router.get("/api/config/options", summary="Config form options")
    def config_options() -> Dict[str, Any]:
        """Return presets, pipeline modes, dataset types, model choices, and defaults for the config builder UI."""
        return create_config_options(provider_factory)

    @router.get("/api/config/schema", summary="Config schema for wizard UI")
    def config_schema() -> Dict[str, Any]:
        """Return structured schema: graph templates → required model fields + compatible datasets,
        datasets → field requirements + default metrics. Used by frontend wizard. (The four former
        pipeline modes are graph templates now; required fields are graph-derived.)"""
        modes: Dict[str, Any] = {}
        for name in sorted(GRAPH_TEMPLATES):
            fields = required_model_fields_for(name)
            if not fields:
                continue
            compatible_datasets = [
                ds_id
                for ds_id in list_registered_datasets()
                if (desc := get_descriptor(ds_id)) and desc.supports_pipeline_mode(name)
            ]
            modes[name] = {
                "required_model_fields": fields,
                "compatible_datasets": compatible_datasets,
            }

        datasets: Dict[str, Any] = {}
        for ds_id in list_registered_datasets():
            desc = get_descriptor(ds_id)
            if desc is None:
                continue
            datasets[ds_id] = {
                "description": desc.description,
                "requires_audio": desc.requires_audio,
                "requires_text": desc.requires_text,
                "supports_generation": desc.supports_generation,
                "evaluation_mode": desc.evaluation_mode,
                "compatible_pipeline_modes": list(desc.compatible_pipeline_modes),
                "required_data_fields": list(desc.required_data_fields),
                "default_metrics": list(desc.default_metrics),
            }

        return {"pipeline_modes": modes, "datasets": datasets}

    @router.post(
        "/api/config/validate",
        summary="Validate config",
        responses={400: {"model": ErrorResponse}},
    )
    def validate_config(payload: EvaluationJobRequest) -> Dict[str, Any]:
        """Validate and normalize an EvaluationConfig dict. Returns 400 on invalid config
        (ConfigurationError → 400 via the app-level handler; a missing optional backend
        lib surfaces as ImportError)."""
        try:
            config = prepare_run_config(
                payload.config, auto_devices=payload.auto_devices
            )
            return {"config": config.to_dict()}
        except ImportError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/api/graph/preview", summary="Preview pipeline DAG")
    def graph_preview_endpoint(payload: EvaluationJobRequest) -> Dict[str, Any]:
        """Return stage graph nodes and levels for the given config (ConfigurationError →
        400 via the app-level handler)."""
        config = load_config(payload.config, auto_devices=payload.auto_devices)
        return graph_preview(config)

    @router.get("/api/graph/nodes", summary="Stage-node catalogue for the builder")
    def graph_nodes_endpoint() -> Dict[str, Any]:
        """The registered node types + I/O contract that the visual builder palette offers (E2)."""
        from ..form_builder import node_catalogue

        return node_catalogue()

    @router.post("/api/graph/node-form", summary="Field-aware form for one operator node")
    def graph_node_form_endpoint(payload: Dict[str, Any]) -> Dict[str, Any]:
        """Re-resolve a node's builder form for its currently-set discriminator fields
        ``{type, params}``: ports + model family + op-specific param switches (so picking
        ``transform.op`` or ``embed.modality`` re-renders the right fields/choices)."""
        from ..form_builder import resolve_node_form

        node_type = payload.get("type") or payload.get("op")
        if not node_type:
            raise HTTPException(status_code=400, detail="node-form needs a 'type'")
        try:
            return resolve_node_form(str(node_type), payload.get("params") or {})
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.get(
        "/api/graph/template/{mode}", summary="Starter graph for a template"
    )
    def graph_template_endpoint(mode: str) -> Dict[str, Any]:
        """A template's default DAG as a canvas seed: nodes (id/type/params), the resolved
        data bindings (artifact + producer — what the auto-wiring derived, so the canvas
        draws *real* edges), and topological levels for layout. The user starts from a
        working graph and only swaps models/params per node."""
        try:
            graph = graph_template(mode)
        except (ValueError, KeyError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        # field-aware contract (ports/label/family/switches) per node so the canvas renders an
        # operator by its discriminator fields — the shared render_node shape.
        return graph_render_payload(graph, lambda n: render_node(n, dict(n.params or {})))

    @router.post("/api/graph/build", summary="Build + validate a canvas graph spec")
    def graph_build_endpoint(payload: Dict[str, Any]) -> Dict[str, Any]:
        """Build the DAG from a builder canvas spec ``{mode, nodes:[{id,type,params}],
        edges:[{from,to}]}`` and return its topological levels — or a 400 with the error (E4).
        ``edges`` ([{from,to}]) is folded into the ``{to: [from,…]}`` shape the builder expects.

        Also returns edit-time *advice* (P3): ``warnings`` (embedding-space mismatches — the same
        check the Run path enforces, surfaced here so the user sees it while editing rather than
        only at submit) and ``metrics`` (which metrics will auto-inject given the graph's produced
        artifacts), so the canvas previews them live.
        """
        from ...pipeline import build_graph_from_spec

        nodes = payload.get("nodes") or []
        mode = payload.get("mode") or "asr_text_retrieval"
        edges: Dict[str, list] = {}
        for e in payload.get("edges") or []:
            if e.get("from") and e.get("to"):
                edges.setdefault(e["to"], []).append(e["from"])
        branches = payload.get("branches") or []
        try:
            if branches:
                # Variant set (P5/§8): expand + CSE-collapse so the response shows
                # the REAL run shape (@branch ids, shared prefix run once).
                from ...pipeline.graph.branches import build_branched_graph

                graph = build_branched_graph(
                    nodes, branches, mode=mode, edges=edges or None
                )
            else:
                graph = build_graph_from_spec(nodes, mode=mode, edges=edges)
        except Exception as exc:  # noqa: BLE001 — surface any build error as 400
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        # Edit-time advice (non-blocking). Embedding-space mismatch is a *warning* here (§4.1 P1,
        # per-node overrides only — no run config) so the user sees it while editing; the run
        # path still hard-validates. Metric-applicability: which metrics land given the graph's
        # produced artifacts (scored + the dataset_source's gt columns).
        from ...config import EvaluationConfig
        from ...evaluation.metric_registry import applicable_metrics
        from ...evaluation.validation import check_embedding_spaces
        from ...pipeline.graph import _effective_outputs

        warnings = check_embedding_spaces(graph, EvaluationConfig())
        produced: set = set()
        for n in graph.nodes:
            produced.update(_effective_outputs(n.stage, n.params))
        metrics = [m.name for m in applicable_metrics(produced, collect_all=True)]
        payload = graph_render_payload(
            graph, lambda n: {"id": n.id, "stage": n.stage, "depends_on": list(n.depends_on)}
        )
        payload["warnings"] = warnings
        payload["metrics"] = metrics
        return payload

    @router.post(
        "/api/graph/from-config",
        summary="Load a YAML config onto the builder canvas (round-trip)",
    )
    def graph_from_config_endpoint(payload: Dict[str, Any]) -> Dict[str, Any]:
        """Parse a node-centric (or legacy) YAML config and return the builder canvas seed for
        its DAG — nodes with their configured models/params + the resolved edges — so an
        existing experiment can be opened in the builder, edited, and re-run. A parse / build
        error is a 400. ``validate=False`` so a partial config still loads for editing.
        """
        import yaml as _yaml

        from evaluator.config import EvaluationConfig
        from evaluator.config.graph_config import to_legacy_dict

        text = payload.get("yaml") or ""
        if not str(text).strip():
            raise HTTPException(status_code=400, detail="Provide a YAML config in 'yaml'.")
        try:
            raw = _yaml.safe_load(text)
            if not isinstance(raw, dict):
                raise ValueError("YAML must be a config mapping (key: value).")
            config = EvaluationConfig.from_dict(to_legacy_dict(raw), validate=False)
            return config_to_canvas_spec(config)
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001 — surface parse/translate/build as 400
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post(
        "/api/graph/to-config",
        summary="Serialize the canvas graph as a runnable config YAML (round-trip out)",
    )
    def graph_to_config_endpoint(payload: Dict[str, Any]) -> Dict[str, Any]:
        """The inverse of /api/graph/from-config: wrap the builder canvas spec as a node-centric
        config (``experiment`` + ``graph`` + optional ``llm``) and return it as YAML — a
        shareable, CLI-runnable artifact, re-loadable via "Load a YAML config". Validated by
        building the DAG (pure topology, no models); an unbuildable graph is a 400.
        """
        import copy

        import yaml as _yaml

        from evaluator.config import EvaluationConfig
        from evaluator.config.graph_config import to_legacy_dict
        from evaluator.pipeline import build_graph_for_config

        from evaluator.webapi.form_builder import lift_single_source_dataset, minimal_edges
        from evaluator.webapi.form_config import _GRAPH_SPEC_KEYS

        spec = payload.get("spec") or {}
        name = str(payload.get("experiment_name") or "builder_export").strip() or "builder_export"
        # deep-copy: lift_single_source_dataset strips the dataset_source node in place
        graph = copy.deepcopy({k: spec[k] for k in _GRAPH_SPEC_KEYS if k in spec})
        if not graph.get("nodes"):
            raise HTTPException(status_code=400, detail="The graph has no nodes to export.")
        # Keep only edges the auto-wiring wouldn't recreate, so bindings don't accumulate across
        # export→load→export and the YAML stays minimal.
        if graph.get("edges"):
            graph["edges"] = minimal_edges(graph["nodes"], graph["edges"])
            if not graph["edges"]:
                del graph["edges"]
        # A single registered-dataset source lifts to a clean top-level `dataset:` block (the node
        # goes structural) so the export reads well and YAML→canvas→YAML is idempotent.
        config: Dict[str, Any] = {"experiment": {"name": name}}
        dataset_block = lift_single_source_dataset(graph.get("nodes") or [])
        if dataset_block:
            config["dataset"] = dataset_block
        config["graph"] = graph
        if spec.get("llm"):
            config["llm"] = spec["llm"]
        try:
            # validate it builds — to_legacy_dict mutates its input, so deep-copy first
            cfg = EvaluationConfig.from_dict(to_legacy_dict(copy.deepcopy(config)), validate=False)
            build_graph_for_config(cfg)
        except Exception as exc:  # noqa: BLE001 — surface any build error as 400
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"yaml": _yaml.safe_dump(config, sort_keys=False)}

    @router.post("/api/sweep/preview", summary="Expand a sweep spec to its run group")
    def sweep_preview_endpoint(payload: Dict[str, Any]) -> Dict[str, Any]:
        """Expand a sweep spec ``{name, base, axes:[{path, values}]}`` to the combination list
        it would launch (Roadmap 4a) — no runs started. Returns the grid + its size; a malformed
        spec is a 400 with the validation error."""
        from ...analysis.sweep import sweep_preview
        from ...errors import ConfigurationError

        try:
            return sweep_preview(payload)
        except ConfigurationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post(
        "/api/config/create",
        summary="Create config from preset + patch",
        responses={400: {"model": ErrorResponse}},
    )
    def create_config(payload: ConfigCreateRequest) -> Dict[str, Any]:
        """Create a new config by merging a patch dict over a preset base
        (ConfigurationError → 400 via the app-level handler)."""
        from evaluator import EvaluationConfig

        if payload.preset_name:
            base_config = EvaluationConfig.from_preset(
                payload.preset_name, validate=False
            )
        else:
            base_config = EvaluationConfig()

        merged = deep_merge_dict(
            nested_config(base_config), dict(payload.config_patch)
        )
        config = EvaluationConfig.from_dict(merged)
        if payload.auto_devices:
            config = config.with_auto_devices()
        return {
            "config": nested_config(config),
            "flat": config.to_dict(),
        }

    @router.post("/api/report/metrics-table", summary="Tidy branch×metric table for a report")
    def report_metrics_table_endpoint(payload: Dict[str, Any]) -> Dict[str, Any]:
        """CLI/web parity (§6): return the flat metrics table + per-query trace count for a
        posted report dict, the shapes a dashboard/dataframe wants from the nested report."""
        from evaluator.analysis.report_export import (
            report_query_traces,
            report_to_metrics_table,
        )

        report = payload.get("report", payload)
        return {
            "metrics": report_to_metrics_table(report),
            "n_traces": len(report_query_traces(report)),
        }

    return router
