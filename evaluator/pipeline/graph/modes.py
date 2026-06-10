"""Named pipeline modes + their derivation into node-id specs.

A named mode (``PIPELINE_MODE_SPECS``) is just an ordered node-id list fed to the
auto-wiring engine. ``_mode_node_ids`` derives that list from the mode + optional-node
flags; ``build_stage_graph`` / ``build_graph_for_config`` are the public entry points.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from .branches import build_branched_graph
from .registry import StageGraph, get_stage_node_def, validate_graph_artifacts
from .wiring import _wire_nodes, build_graph_from_spec


@dataclass(frozen=True)
class PipelineModeSpec:
    mode: str
    required_model_fields: Tuple[str, ...]


def _mode_node_ids(
    mode: str,
    embedding_fusion_enabled: bool,
    query_opt_enabled: bool = False,
    rerank_enabled: bool = False,
    tts_enabled: bool = False,
    sink_enabled: bool = False,
    correction_enabled: bool = False,
) -> List[str]:
    """Ordered node ids for a named mode; auto-wiring derives all edges from these.

    corpus_index is included whenever the mode retrieves. Optional nodes:
    ``query_optimization`` (LLM rewrite/HyDE/multi-query) between ASR and text embedding
    in asr_text_retrieval; ``rerank`` (cross-encoder/MMR/threshold) after retrieval in
    any retrieval mode when refinement is configured."""
    # graph root + optional TTS (synthesize query audio from text) before audio consumers
    src = ["dataset_source", *(["tts"] if tts_enabled else [])]
    sink = ["dataset_sink"] if sink_enabled else []  # terminal leaf, persists outputs
    if mode == "asr_only":
        return [*src, "asr", "metrics", "finalize", *sink]
    # Shared retrieval tail: retrieval (+ optional rerank) -> metrics -> gen -> finalize.
    rerank = ["rerank"] if rerank_enabled else []
    tail = ["retrieval", *rerank, "metrics", "answer_gen", "finalize", *sink]
    if mode == "asr_text_retrieval":
        # correction runs on raw ASR output, before optional query optimization.
        correction = ["query_correction"] if correction_enabled else []
        query = ["query_optimization"] if query_opt_enabled else []
        return [
            *src,
            "corpus_index",
            "asr",
            *correction,
            *query,
            "text_embedding",
            *tail,
        ]
    if mode == "audio_emb_retrieval":
        return [*src, "corpus_index", "audio_embedding", *tail]
    if mode == "audio_text_retrieval":
        if embedding_fusion_enabled:
            return [
                *src,
                "corpus_index",
                "audio_embedding",
                "text_embedding",
                "fusion",
                *tail,
            ]
        return [*src, "corpus_index", "audio_embedding", *tail]
    raise ValueError(f"Unknown pipeline mode: {mode}")


def _required_model_fields(mode: str) -> Tuple[str, ...]:
    """Union of model fields over a mode's nodes (fusion on = the maximal node set)."""
    fields: List[str] = []
    for nid in _mode_node_ids(mode, True):
        field = get_stage_node_def(nid).model_field
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
    rerank_enabled: bool = False,
    tts_enabled: bool = False,
    sink_enabled: bool = False,
    correction_enabled: bool = False,
) -> StageGraph:
    """Build execution DAG for currently supported pipeline modes."""
    resolve_pipeline_mode_spec(mode)  # validates the mode
    node_ids = _mode_node_ids(
        mode,
        embedding_fusion_enabled,
        query_opt_enabled,
        rerank_enabled,
        tts_enabled,
        sink_enabled,
        correction_enabled,
    )
    graph = StageGraph(mode=mode, nodes=_wire_nodes(node_ids))
    validate_graph_artifacts(graph)
    return graph


def _config_rerank_enabled(config: Any) -> bool:
    """Whether a config implies a post-retrieval refine (rerank/MMR/threshold) node."""
    vdb = getattr(config, "vector_db", None)
    if vdb is None:
        return False
    return bool(
        getattr(vdb, "reranker_enabled", False)
        or str(getattr(vdb, "reranker_mode", "none")) != "none"
        or getattr(vdb, "use_mmr", False)
        or getattr(vdb, "min_similarity_threshold", None) is not None
    )


def build_graph_for_config(config: Any) -> StageGraph:
    """Build the execution DAG for a config: explicit ``graph_override`` if present,
    else derived from ``pipeline_mode``. Single source for run + preview + CLI so they
    never disagree. Duck-typed (no config import)."""
    mode = str(config.model.pipeline_mode)
    override = getattr(config, "graph_override", None)
    if override and not override.get("branches"):
        return build_graph_from_spec(
            override["nodes"], mode=mode, edges=override.get("edges")
        )
    fusion = bool(
        getattr(config, "embedding_fusion", None) and config.embedding_fusion.enabled
    )
    query_opt = bool(
        getattr(config, "query_optimization", None)
        and config.query_optimization.enabled
    )
    tts = bool(
        getattr(config, "audio_synthesis", None) and config.audio_synthesis.enabled
    )
    sink = bool(getattr(config, "dataset_sink", None) and config.dataset_sink.enabled)
    correction = bool(
        getattr(config, "query_correction", None) and config.query_correction.enabled
    )
    if override and override.get("branches"):
        base = _mode_node_ids(
            mode,
            fusion,
            query_opt,
            _config_rerank_enabled(config),
            tts,
            sink,
            correction,
        )
        return build_branched_graph(base, override["branches"], mode=mode)
    return build_stage_graph(
        mode,
        embedding_fusion_enabled=fusion,
        query_opt_enabled=query_opt,
        rerank_enabled=_config_rerank_enabled(config),
        tts_enabled=tts,
        sink_enabled=sink,
        correction_enabled=correction,
    )
