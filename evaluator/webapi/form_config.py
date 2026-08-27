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
from typing import Any, Dict, List

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
) -> "tuple[EvaluationConfig, Any]":
    """Translate a builder / Config&Run canvas spec into a fully validated run config — the single
    path shared by ``/api/jobs/from-graph``, ``/ui/run-graph``, and ``/ui/validate-builder`` (so
    the UI, the API, and "Validate" can't drift — "valid" means this exact pipeline succeeds).
    Raises ``ConfigurationError``/``ValueError`` on a missing dataset choice, an unknown node, or an
    embedding-space mismatch (a 400-worthy problem caught *now*, not as a failed job later).
    Returns ``(config, graph)`` — the built graph is returned too (not rebuilt by callers that
    need it, e.g. for edit-time advice) since building it is already required to run the
    embedding-space check."""
    from evaluator.evaluation.validation import validate_graph_embedding_spaces
    from evaluator.pipeline import build_graph_for_config

    _require_dataset_choice(spec)
    config_dict = graph_spec_to_config_dict(spec, experiment_name=experiment_name)
    config = prepare_run_config(config_dict, auto_devices=auto_devices)
    graph = build_graph_for_config(config)
    validate_graph_embedding_spaces(graph, config)
    return config, graph


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


def graph_spec_to_config_dict(
    spec: Dict[str, Any], *, experiment_name: str = "builder_run"
) -> Dict[str, Any]:
    """Translate a builder canvas spec into a legacy ``EvaluationConfig`` dict.

    The canvas exports ``{mode, nodes:[{id,type,params}], edges:[{from,to}], branches?, llm?}``
    — a node-centric ``graph`` block plus an optional global ``llm``. We wrap it and run it
    through :func:`graph_config.build_evaluation_config_kwargs` (the same translator the YAML
    loader uses), so a graph built in the UI takes the identical path as a hand-written config.
    The dataset rides a ``dataset_source`` node whose ``dataset`` param names a registered
    dataset (``build_evaluation_config_kwargs`` synthesizes the ``data.datasets`` entry); a
    graph with no resolvable dataset passes translation but is rejected downstream by
    :func:`prepare_run_config`.
    """
    from evaluator.config.graph_config import build_evaluation_config_kwargs

    # Deep-copy up front: this local `graph` gets mutated below (display-param stripping,
    # derived edges) before being handed off, and the caller's spec must never be altered
    # (safe to call this function twice on the same spec object).
    graph = deepcopy({k: spec[k] for k in _GRAPH_SPEC_KEYS if k in spec})
    # The canvas 'mode' is a display label derived from the nodes — not a config key. Strip it so
    # the graph goes through translation as an explicit-nodes graph (which rejects graph.mode).
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
    # The builder authors only meaningful operations; the derived structural plumbing (the
    # metrics/traces/finalize nodes never drawn on the canvas) is appended by
    # `build_evaluation_config_kwargs` itself now (`pipeline/graph/modes.py:
    # complete_structural_plumbing`, called for every explicit graph — CLI YAML and this UI
    # path alike, one implementation instead of two staying in sync).
    config_dict: Dict[str, Any] = {"graph": graph, "experiment_name": experiment_name}
    if spec.get("llm"):
        config_dict["llm"] = deepcopy(spec["llm"])
    # A drawn feature node (judge / answer-gen / tts / …) enables its capability and carries its
    # params — the loader's _FEATURE_NODE_CONFIG fold handles both from node presence.
    _validate_node_types(graph.get("nodes") or [])
    return build_evaluation_config_kwargs(config_dict)


def _deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively overlay ``overlay`` onto ``base`` (overlay wins; dicts merged)."""
    out = deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out
