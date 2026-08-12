"""Form → validated run-config conversion for the WebAPI.

The config half of the form builder: take a submitted HTML form (flat string dict),
overlay it on the selected preset's full config (config-first — nothing the preset sets
is dropped), and produce a validated :class:`EvaluationConfig`. The UI/option-rendering
half lives in ``form_builder.py`` (which re-exports these names for back-compat).
"""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

from evaluator import ConfigurationError, EvaluationConfig
from evaluator.datasets import validate_dataset_runtime_config
from evaluator.pipeline.factory import check_backend_dependencies


def load_config(
    payload_config: Dict[str, Any], *, auto_devices: bool
) -> EvaluationConfig:
    config = EvaluationConfig.from_dict(payload_config)
    if auto_devices:
        config = config.with_auto_devices()
    return config


#: Dataset path fields a form can set; confined under EVALUATOR_DATA_ROOT when it is set (L1).
_DATASET_PATH_FIELDS = (
    "questions_path", "corpus_path", "prepared_dataset_dir",
    "audio_dir", "data_path", "transcripts_file",
)


def _enforce_data_root(config: EvaluationConfig) -> None:
    """Confine form-supplied dataset paths under ``EVALUATOR_DATA_ROOT`` when it is set (L1).

    Opt-in: with the env unset (the default single-user / local deployment) there is no
    restriction, so absolute local paths keep working. When set — for a shared/multi-tenant
    server — a crafted ``../../etc/...`` that escapes the root is rejected up front.
    """
    root = os.environ.get("EVALUATOR_DATA_ROOT")
    if not root:
        return
    root_resolved = Path(root).resolve()
    data = getattr(config, "data", None)
    for field in _DATASET_PATH_FIELDS:
        value = getattr(data, field, None)
        if not value:
            continue
        target = Path(value).resolve()
        if target != root_resolved and root_resolved not in target.parents:
            raise ConfigurationError(
                f"data.{field} ({value}) is outside the allowed data root {root}"
            )


def prepare_run_config(
    payload_config: Dict[str, Any], *, auto_devices: bool
) -> EvaluationConfig:
    config = load_config(payload_config, auto_devices=auto_devices)
    _enforce_data_root(config)
    check_backend_dependencies(config.vector_db)
    template = config.graph_template
    validate_dataset_runtime_config(
        config,
        retrieval_required=(template != "asr_only"),
    )
    return config


def _require_dataset_choice(spec: Dict[str, Any]) -> None:
    """Reject a builder run whose ``dataset_source`` has no dataset chosen — an empty one would
    silently fall back to the EvaluationConfig default rather than the dataset the user drew."""
    from evaluator.pipeline.graph.operators import node_kind

    for node in spec.get("nodes") or []:
        if not isinstance(node, dict):
            continue
        params = node.get("params") or {}
        if node_kind(node.get("type"), params) == "dataset_source" and not params.get("dataset"):
            raise ValueError("Select a dataset on the dataset-source node before running.")


def build_validated_run_config(
    spec: Dict[str, Any], *, experiment_name: str, auto_devices: bool
) -> EvaluationConfig:
    """Translate a builder / Config&Run canvas spec into a fully validated run config — the single
    path shared by ``/api/jobs/from-graph`` and ``/ui/run-graph`` (so the UI and API can't drift).
    Raises ``ConfigurationError``/``ValueError`` on a missing dataset choice, an unknown node, or an
    embedding-space mismatch (a 400-worthy problem caught *now*, not as a failed job later)."""
    from evaluator.evaluation.validation import validate_graph_embedding_spaces
    from evaluator.pipeline import build_graph_for_config

    _require_dataset_choice(spec)
    config_dict = graph_spec_to_config_dict(spec, experiment_name=experiment_name)
    config = prepare_run_config(config_dict, auto_devices=auto_devices)
    validate_graph_embedding_spaces(build_graph_for_config(config), config)
    return config


#: The graph-block keys a builder canvas spec carries (mode/nodes/edges/branches). Anything
#: else on the spec (e.g. a global ``llm``) is a top-level config block, not a graph key.
_GRAPH_SPEC_KEYS = ("mode", "nodes", "edges", "branches")


def _strip_display_params(nodes: list) -> None:
    """Drop the transient display schema (fields/extra_outputs/suppress_outputs) the canvas
    carries on dataset_source cards — preview metadata, re-derived from the dataset descriptor
    at build; persisting it back would narrow the run's outputs and break edge validation
    (E5). Mutates the node dicts in place."""
    for n in nodes:
        if isinstance(n, dict) and isinstance(n.get("params"), dict):
            for key in ("fields", "extra_outputs", "suppress_outputs"):
                n["params"].pop(key, None)


def _spec_type(spec: Any) -> str:
    """A graph-spec item is a node-type string OR a ``{id, type, params}`` dict."""
    return spec if isinstance(spec, str) else spec.get("type")


def _spec_params(spec: Any) -> dict:
    return {} if isinstance(spec, str) else (spec.get("params") or {})


def _validate_node_types(nodes: list) -> None:
    """Reject an unknown node type up front (so a typo'd type is caught even though the structural
    plumbing the builder *doesn't* draw is auto-derived)."""
    from evaluator.config.graph_config import GraphConfigError
    from evaluator.pipeline.graph.operators import ALIASES, expand_alias, is_operator

    for n in nodes:
        # expand first: a retired-operator spelling (convert{op:asr}) resolves to its
        # first-class node type before the known-name check.
        t, _ = expand_alias(_spec_type(n), _spec_params(n))
        if not (is_operator(t) or t in ALIASES):
            raise GraphConfigError(f"Unknown node type {t!r} in the builder graph.")


def _infer_features(nodes: list) -> Any:
    """Derive a ``FeatureSet`` from which meaningful nodes the user drew — drives which *structural*
    plumbing (which metric comparisons, traces, …) ``_complete_with_plumbing`` appends.

    Only flags reachable from a meaningful palette node are inferred. The FeatureSet's other
    structural-gating flags are intentionally left default because nothing authorable sets them:
    ``sink_enabled`` (sinks ARE structural/derived, never drawn) and a standalone ``trace_enabled``
    (traces ride the judge, which IS inferred). The meaningful-shape flags assemble_specs also reads
    (``hybrid_retrieval``/``rag_rounds``/``refine_*``) only reshape the *kept* retrieve-core nodes,
    so need no re-inference. A future structural node gated on a NEW flag must be added here."""
    from evaluator.pipeline.graph.assembly import FeatureSet
    from evaluator.pipeline.graph.operators import node_kind

    by_kind: Dict[str, dict] = {}
    for n in nodes:
        by_kind.setdefault(node_kind(_spec_type(n), _spec_params(n)), _spec_params(n))
    has = by_kind.__contains__
    return FeatureSet(
        correction_enabled=has("query_correction"),
        query_opt_enabled=has("query_optimization") or has("multi_query_retrieval"),
        query_opt_method=(by_kind.get("query_optimization", {}).get("method")
                          or by_kind.get("multi_query_retrieval", {}).get("method") or "rewrite"),
        rerank_enabled=has("rerank"),
        mmr_enabled=has("mmr"),
        threshold_enabled=has("threshold"),
        answer_gen_enabled=has("answer_gen"),
        judge_enabled=has("answer_judge"),  # assemble_specs adds build_query_traces for the judge
        embedding_fusion_enabled=has("fusion") or has("result_fusion"),
        result_fusion_enabled=has("result_fusion"),
        audio_synthesis_enabled=has("tts"),
    )


def _complete_with_plumbing(legacy: Dict[str, Any], mode: str) -> None:
    """Approach B (UI-graph ↔ execution-DAG split): the builder authors only the *meaningful*
    operations; here we append the *structural* plumbing the template would derive (the metric
    comparisons + report + traces + finalize) so the canvas graph runs as the full DAG. On the
    explicit-wiring
    path the appended nodes' edges are derived here via the authoring wirer (E5); on the legacy
    path they carry no edges and auto-wire. The user's meaningful nodes (incl. per-node models)
    are untouched, so edit-time checks stay exact. No-op on the template-only path (no explicit
    nodes): `assemble_specs` already adds the plumbing there."""
    override = legacy.get("graph_override")
    if not (mode and isinstance(override, dict) and override.get("nodes")):
        return
    from evaluator.pipeline.graph.assembly import assemble_specs
    from evaluator.pipeline.graph.operators import node_kind
    from evaluator.pipeline.graph.registry import is_structural

    nodes = override["nodes"]
    have = {node_kind(_spec_type(n), _spec_params(n)) for n in nodes}
    appended = []
    for spec in assemble_specs(mode, _infer_features(nodes)):
        st, sp = _spec_type(spec), _spec_params(spec)
        if is_structural(st, sp) and node_kind(st, sp) not in have:
            nodes.append(spec)
            appended.append(spec if isinstance(spec, str) else (spec.get("id") or st))
            have.add(node_kind(st, sp))
    # E5: with explicit port wiring the appended plumbing needs its edges too — derive them
    # via the authoring wirer over the FULL list and keep those touching an appended node
    # (the user's drawn edges stay authoritative for the meaningful graph).
    edges = override.get("edges")
    from evaluator.pipeline.graph.wiring import _has_port_edges

    if appended and _has_port_edges(edges):
        from evaluator.pipeline.graph.wiring import emit_edges

        app_ids = set(appended)

        # `output` may be omitted when it equals `input` — compare on the resolved pair so the
        # short and long spellings of one edge dedup against each other.
        def _key(e):
            return (e.get("from"), e.get("output", e.get("input")),
                    e.get("to"), e.get("input"))

        existing = {_key(e) for e in edges}
        derived = [
            e for e in emit_edges(nodes)
            if (e["to"] in app_ids or e["from"] in app_ids) and _key(e) not in existing
        ]
        override["edges"] = list(edges) + derived


def graph_spec_to_config_dict(
    spec: Dict[str, Any], *, experiment_name: str = "builder_run"
) -> Dict[str, Any]:
    """Translate a builder canvas spec into a legacy ``EvaluationConfig`` dict.

    The canvas exports ``{mode, nodes:[{id,type,params}], edges:[{from,to}], branches?, llm?}``
    — a node-centric ``graph`` block plus an optional global ``llm``. We wrap it and run it
    through :func:`graph_config.to_legacy_dict` (the same translator the YAML loader uses), so
    a graph built in the UI takes the identical path as a hand-written config. The dataset
    rides a ``dataset_source`` node whose ``dataset`` param names a registered dataset
    (``to_legacy_dict`` synthesizes the ``data.datasets`` entry); a graph with no resolvable
    dataset passes translation but is rejected downstream by :func:`prepare_run_config`.
    """
    from evaluator.config.graph_config import to_legacy_dict

    # to_legacy_dict mutates the node dicts it's handed (it strips folded params in place);
    # deep-copy so the caller's spec is never altered (safe to call twice on the same object).
    graph = deepcopy({k: spec[k] for k in _GRAPH_SPEC_KEYS if k in spec})
    # The canvas 'mode' is a display label derived from the nodes — not a config key. Strip it so
    # the graph goes through to_legacy_dict as an explicit-nodes graph (which rejects graph.mode).
    graph.pop("mode", None)
    _strip_display_params(graph.get("nodes") or [])
    # The builder IS the authoring assistant: a spec whose edges aren't port-level yet
    # (palette drafts, legacy saved graphs with ordering-only edges) gets its edge set
    # derived here — the same emit_edges the CLI offers; canvas port edges pass through
    # untouched and stay authoritative.
    from evaluator.pipeline.graph.wiring import _has_port_edges, emit_edges

    if graph.get("nodes") and not _has_port_edges(graph.get("edges")):
        ordering = [e for e in (graph.get("edges") or []) if isinstance(e, dict)]
        graph["edges"] = emit_edges(graph["nodes"]) + ordering
    config_dict: Dict[str, Any] = {"graph": graph, "experiment_name": experiment_name}
    if spec.get("llm"):
        config_dict["llm"] = deepcopy(spec["llm"])
    # A drawn feature node (judge / answer-gen / tts / …) enables its capability and carries its
    # params — the loader's _FEATURE_NODE_CONFIG fold handles both from node presence.
    _validate_node_types(graph.get("nodes") or [])
    legacy = to_legacy_dict(config_dict)
    # The builder authors only meaningful operations; append the derived structural plumbing so the
    # canvas graph runs as the full execution DAG (skipped for branched graphs: they expand apart).
    if not graph.get("branches"):
        from evaluator.pipeline.graph.modes import label_from_graph

        # the plumbing template is derived from the drawn nodes (the label), not a config mode.
        _complete_with_plumbing(legacy, label_from_graph(legacy.get("graph_override")))
    return legacy


def _deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively overlay ``overlay`` onto ``base`` (overlay wins; dicts merged)."""
    out = deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out
