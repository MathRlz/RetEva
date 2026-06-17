"""Named pipeline modes + their derivation into node-id specs.

A named mode (``PIPELINE_MODE_SPECS``) is just an ordered node-id list fed to the
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
    get_stage_node_def,
    node_model_field,
    validate_graph_artifacts,
)
from .wiring import _normalize_spec_item, _wire_nodes, build_graph_from_spec


@dataclass(frozen=True)
class PipelineModeSpec:
    mode: str
    required_model_fields: Tuple[str, ...]


def _required_model_fields(mode: str) -> Tuple[str, ...]:
    """Union of model fields over a mode's nodes (maximal feature set)."""
    fields: List[str] = []
    for spec in assemble_specs(mode, FeatureSet.maximal()):
        _id, ntype, _params = _normalize_spec_item(spec)
        field = node_model_field(ntype, _params)  # resolves a callable model_field
        if field and field not in fields:
            fields.append(field)
    return tuple(fields)


def _make_spec(mode: str) -> PipelineModeSpec:
    return PipelineModeSpec(
        mode=mode,
        required_model_fields=_required_model_fields(mode),
    )


PIPELINE_MODE_SPECS: Dict[str, PipelineModeSpec] = {
    mode: _make_spec(mode)
    for mode in (
        "asr_only",
        "asr_text_retrieval",
        "audio_emb_retrieval",
        "audio_text_retrieval",
    )
}


def list_pipeline_mode_specs() -> List[PipelineModeSpec]:
    return [PIPELINE_MODE_SPECS[key] for key in sorted(PIPELINE_MODE_SPECS.keys())]


def resolve_pipeline_mode_spec(mode: str) -> PipelineModeSpec:
    if mode not in PIPELINE_MODE_SPECS:
        available = ", ".join(sorted(PIPELINE_MODE_SPECS.keys()))
        raise ValueError(f"Unknown pipeline mode: {mode}. Available modes: {available}")
    return PIPELINE_MODE_SPECS[mode]


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
    tts_enabled: bool = False,
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
    resolve_pipeline_mode_spec(mode)  # validates the mode
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
        tts_enabled=tts_enabled,
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


def build_graph_for_config(config: Any) -> StageGraph:
    """Build the execution DAG for a config: explicit ``graph_override`` if present,
    else derived from ``pipeline_mode``. Single source for run + preview + CLI so they
    never disagree. Duck-typed (no config import)."""
    mode = str(config.model.pipeline_mode)
    override = getattr(config, "graph_override", None)
    if override and not override.get("branches"):
        return build_graph_from_spec(
            _attach_dataset_fields(override["nodes"], config),
            mode=mode,
            edges=override.get("edges"),
        )
    base = _attach_dataset_fields(
        assemble_specs(mode, _features_from_config(config)), config
    )
    if override and override.get("branches"):
        return build_branched_graph(base, override["branches"], mode=mode)
    return build_graph_from_spec(base, mode=mode)


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
        tts_enabled=bool(
            getattr(config, "audio_synthesis", None) and config.audio_synthesis.enabled
        ),
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
