"""Per-stage model offload planning (model lifecycle).

Moved verbatim from the former ``evaluation/phased.py`` (Phase 1, X4). Maps each stage
position to the models whose last use falls there, so the executor can release them
mid-run (frees the device) once a model is no longer needed.
"""

from __future__ import annotations

from typing import Dict

from .state import RunState

# Stage id -> the RunState attribute holding the pipeline whose model that stage runs.
_STAGE_PIPELINE_ATTR = {
    "asr": "asr_pipeline",
    "text_embedding": "text_embedding_pipeline",
    "audio_embedding": "audio_embedding_pipeline",
}


def _stage_model(state: "RunState", stage: str):
    """The model a stage runs, or None for model-free stages (fusion/metrics).

    The rerank stage's model is its cross-encoder reranker (when configured); it is used
    only there, so it can be freed once reranking completes.
    """
    if stage == "rerank":
        rp = getattr(state, "retrieval_pipeline", None)
        return getattr(rp, "reranker", None) if rp is not None else None
    if stage == "corpus_index":
        # Corpus embedding uses the same embedder instance as the query side (text
        # preferred, else audio); returning it lets the offload planner free that model
        # only after the LATER of the corpus and query embedding stages.
        tep = getattr(state, "text_embedding_pipeline", None)
        model = getattr(tep, "model", None) if tep is not None else None
        if model is not None:
            return model
        aep = getattr(state, "audio_embedding_pipeline", None)
        return getattr(aep, "model", None) if aep is not None else None
    attr = _STAGE_PIPELINE_ATTR.get(stage)
    pipe = getattr(state, attr, None) if attr else None
    return getattr(pipe, "model", None) if pipe is not None else None


def _plan_stage_offloads(state: "RunState", flat_nodes, query_opt_enabled: bool):
    """Map each stage position to the models whose LAST use is there (so they can be freed).

    Keyed by model *instance* so a model shared across stages (e.g. a multimodal embedder
    used for both audio and text) is released only after its final stage. The text
    embedder is excluded when query optimization is on, since query-opt may re-embed.
    """
    if not state.offload_after_stage or state.service_provider is None:
        return {}
    excluded = set()
    if query_opt_enabled:
        te = _stage_model(state, "text_embedding")
        if te is not None:
            excluded.add(id(te))
    last: Dict[int, tuple] = {}
    for pos, node in enumerate(flat_nodes):
        model = _stage_model(state, node.stage)
        if model is not None and id(model) not in excluded:
            last[id(model)] = (pos, model)
    by_pos: Dict[int, list] = {}
    for pos, model in last.values():
        by_pos.setdefault(pos, []).append(model)
    return by_pos
