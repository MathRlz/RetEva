"""Named pipeline modes + their derivation into node-id specs.

A named mode (``GRAPH_TEMPLATE_SPECS``) is just an ordered node-id list fed to the
auto-wiring engine. The list is assembled declaratively from a :class:`FeatureSet`
(`graph/assembly.py`); ``build_stage_graph`` / ``build_graph_for_config`` are the public
entry points (the former for tests, the latter the single config chokepoint).
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .assembly import FeatureSet, assemble_specs
from .branches import build_branched_graph
from .registry import (
    StageGraph,
    node_model_field,
    validate_graph_artifacts,
)
from .wiring import _normalize_spec_item, _wire_nodes, build_graph_from_spec


@dataclass(frozen=True)
class GraphTemplateSpec:
    name: str
    required_model_fields: Tuple[str, ...]


def _required_model_fields(name: str) -> Tuple[str, ...]:
    """Union of model fields over a template's nodes (maximal feature set)."""
    fields: List[str] = []
    for spec in assemble_specs(name, FeatureSet.maximal()):
        _id, ntype, _params = _normalize_spec_item(spec)
        field = node_model_field(ntype, _params)  # resolves a callable model_field
        if field and field not in fields:
            fields.append(field)
    return tuple(fields)


def _make_template_spec(name: str) -> GraphTemplateSpec:
    return GraphTemplateSpec(
        name=name,
        required_model_fields=_required_model_fields(name),
    )


GRAPH_TEMPLATE_SPECS: Dict[str, GraphTemplateSpec] = {
    name: _make_template_spec(name)
    for name in (
        "asr_only",
        "asr_text_retrieval",
        "audio_emb_retrieval",
        "audio_text_retrieval",
    )
}


def resolve_graph_template(name: str) -> GraphTemplateSpec:
    if name not in GRAPH_TEMPLATE_SPECS:
        available = ", ".join(sorted(GRAPH_TEMPLATE_SPECS.keys()))
        raise ValueError(f"Unknown graph template: {name}. Available templates: {available}")
    return GRAPH_TEMPLATE_SPECS[name]


def build_stage_graph(
    mode: str,
    *,
    embedding_fusion_enabled: bool = False,
    query_opt_enabled: bool = False,
    hybrid_retrieval: bool = False,
    rerank_enabled: bool = False,
    mmr_enabled: bool = False,
    threshold_enabled: bool = False,
    refine_ops: tuple = (),
    sink_enabled: bool = False,
    correction_enabled: bool = False,
    answer_gen_enabled: bool = False,
    judge_enabled: bool = False,
    trace_enabled: bool = False,
    result_fusion_enabled: bool = False,
    rag_rounds: int = 1,
    refine_method: str = "rewrite_with_context",
    refine_context_top_k: int = 3,
    query_opt_method: str = "rewrite",
) -> StageGraph:
    """Build execution DAG for currently supported pipeline modes (kwargs → FeatureSet)."""
    resolve_graph_template(mode)  # validates the mode
    features = FeatureSet(
        embedding_fusion_enabled=embedding_fusion_enabled,
        result_fusion_enabled=result_fusion_enabled,
        query_opt_enabled=query_opt_enabled,
        query_opt_method=query_opt_method,
        hybrid_retrieval=hybrid_retrieval,
        rerank_enabled=rerank_enabled,
        mmr_enabled=mmr_enabled,
        threshold_enabled=threshold_enabled,
        refine_ops=tuple(refine_ops),
        sink_enabled=sink_enabled,
        correction_enabled=correction_enabled,
        answer_gen_enabled=answer_gen_enabled,
        judge_enabled=judge_enabled,
        trace_enabled=trace_enabled,
        rag_rounds=rag_rounds,
        refine_method=refine_method,
        refine_context_top_k=refine_context_top_k,
    )
    graph = StageGraph(mode=mode, nodes=_wire_nodes(assemble_specs(mode, features)))
    validate_graph_artifacts(graph)
    return graph


def _dataset_fields_for(config: Any, params: Optional[dict]) -> Optional[dict]:
    """The declared column schema for a dataset_source node instance.

    Multi-source nodes (``params.dataset``) resolve their `datasets:` entry overlaid on
    ``config.data`` (same overlay as ``load_runtime_datasets``); single-source resolves
    the global ``config.data``. Unresolvable → None (the node keeps the static outputs).
    """
    from dataclasses import replace

    from ...datasets.descriptor import resolve_dataset_descriptor

    data = getattr(config, "data", None)
    if data is None:
        return None
    try:
        ds_id = (params or {}).get("dataset")
        if ds_id:
            entry = (getattr(data, "datasets", None) or {}).get(str(ds_id)) or {}
            overlay = {k: v for k, v in entry.items() if k not in ("role", "datasets")}
            data = replace(data, datasets=None, **overlay)
        descriptor = resolve_dataset_descriptor(data)
    except Exception:
        return None
    if not descriptor.fields:
        return None
    out = {"fields": dict(descriptor.fields)}
    if descriptor.embedding_space:
        out["embedding_space"] = descriptor.embedding_space
    return out


def _attach_dataset_fields(node_spec: Any, config: Any) -> list:
    """Inject ``params.fields`` (the column schema) into dataset_source spec entries.

    Runs before wiring so ``_effective_outputs`` (and every DAG display surface) sees
    per-dataset columns. Entries may be node-type strings or {id, type, params} dicts.
    """
    out = []
    for entry in node_spec:
        if isinstance(entry, str):
            stage, spec = entry, {"id": entry, "type": entry}
        else:
            spec = dict(entry)
            stage = spec.get("type") or spec.get("id")
        if stage != "dataset_source":
            out.append(entry)
            continue
        params = dict(spec.get("params") or {})
        injected = _dataset_fields_for(config, params)
        if injected and "fields" not in params:
            params.update(injected)
            spec["params"] = params
            out.append(spec)
        else:
            out.append(entry)
    return out


def _wire_mode_graph(
    mode: Optional[str],
    features: "FeatureSet",
    *,
    graph_override: Optional[dict],
    config: Any,
    attach_fields: bool,
) -> StageGraph:
    """Shared assembly tail for both graph builders: an explicit ``graph_override`` if given,
    else ``assemble_specs(mode, features)``; branch-expand or wire. ``attach_fields`` injects the
    dataset column schema into ``dataset_source`` nodes — done for the config/preview path (so the
    DAG display shows real columns) but NOT for the run (which wires from the static node outputs,
    the source of the run's reference_transcription binding). A mode-less explicit DAG labels
    "custom"."""
    override = graph_override or {}
    if override.get("nodes") and not override.get("branches"):
        nodes = override["nodes"]
        if attach_fields:
            nodes = _attach_dataset_fields(nodes, config)
        return build_graph_from_spec(
            nodes, mode=mode or "custom", edges=override.get("edges")
        )
    if mode is None:
        raise ValueError(
            "a config needs a template (graph.mode) or an explicit graph.nodes to build from."
        )
    base = assemble_specs(mode, features)
    if attach_fields:
        base = _attach_dataset_fields(base, config)
    if override.get("branches"):
        return build_branched_graph(base, override["branches"], mode=mode)
    return build_graph_from_spec(base, mode=mode)


def _config_template(config: Any) -> Optional[str]:
    """The graph template a config selects (``graph_override['template']``, set from
    ``graph.mode``). ``None`` for an explicit-graph config — the run derives the label instead."""
    override = getattr(config, "graph_override", None) or {}
    template = override.get("template")
    return str(template) if template else None


def build_graph_for_config(config: Any) -> StageGraph:
    """Build the execution DAG for a config: an explicit ``graph.nodes`` if present, else the
    config's graph *template* (``graph.mode``) expanded with its feature flags. Single source for
    preview + CLI + the factory's build plan. Duck-typed (no config import). The run uses
    :func:`build_run_graph`, which sources its feature flags from the built pipelines instead."""
    return _wire_mode_graph(
        _config_template(config),
        _features_from_config(config),
        graph_override=getattr(config, "graph_override", None),
        config=config,
        attach_fields=True,
    )


def build_run_graph(
    mode,
    *,
    graph_override,
    embedding_fusion_config,
    query_opt_config,
    retrieval_pipeline,
    eval_config,
    query_correction_config=None,
    trace_limit=0,
):
    """Build the execution DAG for a run: like :func:`build_graph_for_config` but the
    feature flags are sourced from what actually got built/bound at runtime — fusion /
    query-opt / correction from the run's ``RunFeatures``, rerank / mmr / threshold off the
    built retrieval pipeline's strategy config, ``trace`` from the run's trace limit — so the
    graph reflects reality, not just the config's declared intent. (Moved from the former
    ``pipeline/run_graph.py``; shares ``_wire_mode_graph`` with the config builder.)"""
    from dataclasses import replace

    def _enabled(cfg):
        return bool(cfg is not None and getattr(cfg, "enabled", False))

    fusion_on = _enabled(embedding_fusion_config)
    # The built retrieval pipeline is the authoritative source for the refine sub-steps
    # (rerank / mmr / threshold) — read them off its strategy config.
    sc = getattr(retrieval_pipeline, "strategy_config", None)
    rerank_on = bool(
        sc
        and (
            str(sc.reranking.mode) != "none"
            or getattr(retrieval_pipeline, "reranker", None) is not None
        )
    )
    mmr_on = bool(sc and sc.post_processing.use_mmr)
    threshold_on = bool(sc and sc.post_processing.min_similarity_threshold is not None)
    features = replace(
        _features_from_config(eval_config),
        embedding_fusion_enabled=fusion_on,
        result_fusion_enabled=fusion_on
        and getattr(embedding_fusion_config, "level", "embedding") == "result",
        query_opt_enabled=_enabled(query_opt_config),
        query_opt_method=str(getattr(query_opt_config, "method", "rewrite")),
        correction_enabled=_enabled(query_correction_config),
        rerank_enabled=rerank_on,
        mmr_enabled=mmr_on,
        threshold_enabled=threshold_on,
        trace_enabled=trace_limit > 0,
    )
    # attach_fields=False — the run wires from the static dataset_source outputs (parity).
    return _wire_mode_graph(
        mode, features, graph_override=graph_override, config=eval_config, attach_fields=False
    )


def _features_from_config(config: Any) -> FeatureSet:
    """The single place config → FeatureSet: each optional capability's `enabled` flag +
    the structural choices (fusion level, query-opt method, rag rounds)."""
    fusion = bool(
        getattr(config, "embedding_fusion", None) and config.embedding_fusion.enabled
    )
    rag = getattr(config, "rag", None)
    _vdb = getattr(config, "vector_db", None)
    return FeatureSet(
        embedding_fusion_enabled=fusion,
        result_fusion_enabled=fusion
        and getattr(config.embedding_fusion, "level", "embedding") == "result",
        query_opt_enabled=bool(
            getattr(config, "query_optimization", None)
            and config.query_optimization.enabled
        ),
        query_opt_method=str(
            getattr(getattr(config, "query_optimization", None), "method", "rewrite")
        ),
        hybrid_retrieval=bool(
            _vdb and str(getattr(_vdb, "retrieval_mode", "dense")) == "hybrid"
        ),
        rerank_enabled=bool(
            _vdb
            and (
                getattr(_vdb, "reranker_enabled", False)
                or str(getattr(_vdb, "reranker_mode", "none")) != "none"
            )
        ),
        mmr_enabled=bool(_vdb and getattr(_vdb, "use_mmr", False)),
        threshold_enabled=bool(
            _vdb and getattr(_vdb, "min_similarity_threshold", None) is not None
        ),
        refine_ops=tuple(getattr(_vdb, "refine_ops", None) or ()),
        sink_enabled=bool(
            getattr(config, "dataset_sink", None) and config.dataset_sink.enabled
        ),
        correction_enabled=bool(
            getattr(config, "query_correction", None)
            and config.query_correction.enabled
        ),
        answer_gen_enabled=bool(
            getattr(config, "answer_generation", None)
            and getattr(config.answer_generation, "enabled", False)
        ),
        judge_enabled=bool(
            getattr(config, "judge", None) and getattr(config.judge, "enabled", False)
        ),
        trace_enabled=int(getattr(getattr(config, "data", None), "trace_limit", 0) or 0) > 0,
        rag_rounds=int(getattr(rag, "rounds", 1) or 1),
        refine_method=str(getattr(rag, "refine_method", "rewrite_with_context")),
        refine_context_top_k=int(getattr(rag, "refine_context_top_k", 3) or 3),
    )
