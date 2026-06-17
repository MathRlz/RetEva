"""Config-related WebAPI endpoints."""

from typing import Any, Callable, Dict

from fastapi import APIRouter, HTTPException

from evaluator.pipeline.stage_graph import (
    PIPELINE_MODE_SPECS,
    resolve_pipeline_mode_spec,
)
from evaluator.datasets.descriptor import list_registered_datasets, get_descriptor
from evaluator.services import ModelServiceProvider
from evaluator.webapi.form_builder import (
    create_config_options,
    deep_merge_dict,
    graph_preview,
    load_config,
    nested_config,
    prepare_run_config,
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
        """Return structured schema: pipeline modes → required model fields + compatible datasets,
        datasets → field requirements + default metrics. Used by frontend wizard."""
        modes: Dict[str, Any] = {}
        for mode in sorted(PIPELINE_MODE_SPECS.keys()):
            try:
                spec = resolve_pipeline_mode_spec(mode)
            except ValueError:
                continue
            compatible_datasets = [
                ds_id
                for ds_id in list_registered_datasets()
                if (desc := get_descriptor(ds_id)) and desc.supports_pipeline_mode(mode)
            ]
            modes[mode] = {
                "required_model_fields": list(spec.required_model_fields),
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
        "/api/graph/template/{mode}", summary="Starter graph for a pipeline mode"
    )
    def graph_template_endpoint(mode: str) -> Dict[str, Any]:
        """The mode's default DAG as a canvas seed: nodes (id/type/params), the resolved
        data bindings (artifact + producer — what the auto-wiring derived, so the canvas
        draws *real* edges), and topological levels for layout. The user starts from a
        working graph and only swaps models/params per node."""
        from ...pipeline.stage_graph import build_stage_graph
        from ..form_builder import resolve_node_form

        try:
            graph = build_stage_graph(
                mode,
                # audio_text is the fusion mode — its starter includes the fusion path.
                embedding_fusion_enabled=(mode == "audio_text_retrieval"),
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {
            "mode": graph.mode,
            "levels": [[n.id for n in level] for level in graph.topological_levels()],
            "nodes": [
                {
                    "id": n.id,
                    "type": n.stage,
                    "params": dict(n.params or {}),
                    "bindings": [list(b) for b in n.bindings],
                    # field-aware contract (ports/label/family/switches) for THIS instance,
                    # so the canvas renders an operator node by its discriminator fields.
                    **resolve_node_form(n.stage, dict(n.params or {})),
                }
                for n in graph.nodes
            ],
        }

    @router.post("/api/graph/build", summary="Build + validate a canvas graph spec")
    def graph_build_endpoint(payload: Dict[str, Any]) -> Dict[str, Any]:
        """Build the DAG from a builder canvas spec ``{mode, nodes:[{id,type,params}],
        edges:[{from,to}]}`` and return its topological levels — or a 400 with the error (E4).
        ``edges`` ([{from,to}]) is folded into the ``{to: [from,…]}`` shape the builder expects.
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
            # Per-node V[s] check (§4.1 P1) — per-node embedder overrides only
            # (no run config here, so global fields contribute nothing).
            from ...config import EvaluationConfig
            from ...evaluation.validation import validate_graph_embedding_spaces

            validate_graph_embedding_spaces(graph, EvaluationConfig())
        except Exception as exc:  # noqa: BLE001 — surface any build error as 400
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "mode": graph.mode,
            "levels": [[n.id for n in level] for level in graph.topological_levels()],
            "nodes": [
                {"id": n.id, "stage": n.stage, "depends_on": list(n.depends_on)}
                for n in graph.nodes
            ],
        }

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
