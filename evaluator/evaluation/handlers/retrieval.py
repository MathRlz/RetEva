"""Retrieval stage handlers: candidate generation + rerank/refinement.

Each handler
registers itself via ``@register_stage_handler`` at import time.
"""

from __future__ import annotations

import numpy as np

from ..stage_registry import register_stage_handler
from ...logging_config import get_logger, TimingContext
from ..helpers import _search_results_to_keys
from ..item_isolation import isolate_batch
from ..executor.state import RunState
from ..executor.node_pipeline import _node_reranking
from ._common import publish_keyed_or_plain, is_asr_text_retrieval
from .retrieval_debug import log_retrieval_debug as _log_retrieval_debug

logger = get_logger(__name__)


def _publish_retrieved(s: RunState, results_list: list, query_ids: list) -> None:
    """Publish ``retrieved`` (keyed when ids align 1:1, else plain) via the shared publish
    contract — so rerank/metrics/answer_gen read the list, metric nodes read it keyed."""
    publish_keyed_or_plain(s, "retrieved", results_list, query_ids)


def _retrieval_query_texts(s: RunState, rp=None):
    """Query texts feeding retrieval/rerank (mode-dependent); None for audio_emb.

    Bus-only since M1d-2: the effective query text (asr modes) / spoken reference
    (fusion mode) come from the ctx. ``rp`` is the pipeline actually used (the
    vector_index artifact); falls back to the shared one."""
    rp = rp if rp is not None else s.retrieval_pipeline
    if is_asr_text_retrieval(s):
        # the effective (most-processed) query for sparse/hybrid scoring (QUERY_TEXT_CHAIN)
        return s.input("query_text", default=None)
    if s.audio_embedding_pipeline is not None:
        # Audio query: the spoken reference can drive sparse/hybrid lexical scoring only in
        # cross-modal audio↔text retrieval; pure audio-embedding self-retrieval has no text
        # query and supports dense only. The audio_emb/audio_text split is a run policy the
        # graph can't express (identical node sets), so it reads the run's mode label.
        if s.mode == "audio_text_retrieval":
            return s.get_artifact("reference_transcription", default=None)
        if rp.strategy_config.core.mode != "dense":
            raise ValueError(
                "audio_emb_retrieval supports only retrieval_mode='dense'. "
                "Sparse/hybrid requires query text unavailable in this path."
            )
    return None


def _finalize_retrieval(s: RunState, results_list, query_vectors, query_ids, rp=None) -> None:
    """Publish final (payload, score) results to the bus; log debug.

    Bus-only since M1c-2: downstream (metrics / answer_gen / finalize / aggregate)
    reads the `retrieved` artifact via `_retrieved_from_bus`, not RunState mirrors."""
    retrieved_keys = [_search_results_to_keys(results) for results in results_list]
    # Publish the `retrieved` artifact (list of per-query (payload, score) lists) under
    # this node's id so downstream reads the right producer (retrieval or a rerank). Same
    # shape as retrieve_candidates output, so rerank instances chain (R4c/D2). Keyed as an
    # ItemSet (W3) when query ids align — get_artifact unwraps to the list for legacy
    # consumers; metric nodes read it keyed via get_items.
    _publish_retrieved(s, results_list, query_ids)
    logger.info(f"Retrieval complete: {len(retrieved_keys)} queries")
    _log_retrieval_debug(
        rp if rp is not None else s.retrieval_pipeline,
        retrieved_keys,
        list(s.get_artifact("reference_transcription", default=[])),
        list(s.get_artifact("relevant_docs", default=[])),
        list(s.input("query_text", default=[])),
        query_ids,
        query_vectors,
        s.mode,
        s.k,
    )


@register_stage_handler("index", time_key="retrieval_s")
def _stage_index(s: RunState) -> None:
    """Build the per-node vector store + index from ``corpus_vectors`` (§4 split).

    The node's params pick the backend (``store`` + backend essentials); defaults
    come from the global ``vector_db`` config. Publishes the indexed pipeline as
    this node's ``vector_index`` — retrieval reads it artifact-first, so two
    ``vector_db`` nodes with different backends coexist across branches.
    """
    cv = s.get_artifact("corpus_vectors", default=None)
    if cv is None or s.config is None:
        return
    from ...models.retrieval.strategy import RetrievalStrategyConfig
    from ...pipeline.factory import (
        create_vector_store_from_config,
        effective_vector_db_config,
    )
    from ...pipeline.retrieval_pipeline import RetrievalPipeline
    from ...services.corpus_index import build_index_from_vectors

    params = s.node_params
    effective = effective_vector_db_config(s.config.vector_db, params)
    vectors = np.asarray(cv.vectors)
    embedding_dim = int(vectors.shape[1]) if vectors.ndim == 2 else None
    store = create_vector_store_from_config(effective, embedding_dim=embedding_dim)
    rp = RetrievalPipeline(
        store,
        s.cache_manager,
        strategy_config=RetrievalStrategyConfig.from_vector_db_config(effective),
        # The shared cross-encoder (built once by the factory) rides along so a
        # following rerank node refines with the configured model — same instance,
        # so the offload planner's free-after-last-use semantics are unchanged.
        reranker=getattr(getattr(s, "retrieval_pipeline", None), "reranker", None),
    )
    build_index_from_vectors(rp, cv)
    rp.embedding_space = cv.space  # space tag rides the index (§4 V[s] typing)
    s.put_artifact("vector_index", rp)


@register_stage_handler("search", time_key="retrieval_s")
def _stage_search(s: RunState) -> None:
    """The ``search`` operator: plain retrieval, or the multi-query/decompose fan-out
    composite (which lives in the query handlers). Bodies unchanged."""
    from .query import _stage_multi_query_retrieval
    from ._dispatch import dispatch_operator

    return dispatch_operator("search", {
        "retrieval": _stage_retrieval,
        "multi_query_retrieval": _stage_multi_query_retrieval,
    }, s)


def _stage_retrieval(s: RunState) -> None:
    """Candidate generation (dense/sparse/hybrid). When a ``rerank`` node follows
    (refinement configured) this only fetches candidates; otherwise it finalizes."""
    # The retrieval-input embeddings: the highest-priority published stream resolved by
    # s.input (one_of fused > audio > text > precomputed). A result-fusion retrieval
    # instance pins ONE stream via the `vectors` param (so two retrievals over the audio
    # and text streams feed result_fusion). Bus-only (M1d): a missing producer is a graph bug.
    params = s.node_params
    vname = params.get("vectors")
    query_vectors = s.get_artifact(vname) if vname else s.input("query_vectors")
    if query_vectors is None:
        from ...errors import ConfigurationError

        raise ConfigurationError(
            "retrieval got no query vectors: no upstream embedder published "
            f"'{vname or 'query_vectors'}' (audio/text/fused). Likely a graph-wiring bug "
            "(a missing/misordered embedding node) or a stale cache — try "
            "`evaluator cache clear` and re-run."
        )
    if isinstance(query_vectors, list) and len(query_vectors) > 0:
        query_vectors = np.array(query_vectors)
    s.cb("phase_3_retrieval", 0, s.total, "Phase 3: Retrieval")
    # The vector_index artifact is the indexed retrieval pipeline (published by
    # vector_db); fall back to the shared pipeline for direct callers (R4a).
    rp = s.get_artifact("vector_index", default=s.retrieval_pipeline)
    # Runtime space guard (2b): loud error if the bound query stream and the index live in
    # incompatible embedding spaces. No-op when either space is unknown (defense-in-depth
    # behind the pre-flight validators). Only the vector-dotting arms (dense/hybrid) qualify.
    cfg_mode = str(getattr(getattr(rp, "strategy_config", None) and rp.strategy_config.core,
                           "mode", "dense"))
    if (params.get("mode") or cfg_mode) != "sparse" and hasattr(rp, "assert_query_space"):
        from ..validation import resolve_query_space

        rp.assert_query_space(resolve_query_space(s.config, vname))
    # Per-node k (R2): a branch may retrieve a different depth.
    k = int(params.get("k", s.k))
    # Per-node retrieval mode (R-hybrid): the hybrid DAG runs its dense + sparse arms as two
    # retrieval nodes (mode=dense / mode=sparse) feeding result_fusion, instead of the
    # monolithic in-pipeline hybrid fuse. Absent ⇒ the pipeline's configured mode.
    rmode = params.get("mode")
    node_id = getattr(s.current_node, "id", "retrieval")
    with TimingContext("Retrieval Phase", logger):
        qtexts = _retrieval_query_texts(s, rp)
        nq = len(query_vectors)
        # Per-item identity rides the keyed query_vectors ItemSet (M1d-2).
        keyed = s.keyed_items(vname) if vname else s.input_items("query_vectors")
        ids = (
            [str(i) for i in keyed.ids]
            if keyed is not None
            else [str(i) for i in range(nq)]
        )

        def _row_texts(i: int):
            return [qtexts[i]] if qtexts is not None else None

        if rp.needs_refinement or s.refine_in_graph or getattr(s, "fuse_in_graph", False):
            # Per-item isolation (T1): one query failing to fetch candidates drops out (empty
            # candidate list, recorded) instead of aborting; keyed report excludes it.
            candidates = isolate_batch(
                ids,
                list(range(nq)),
                batch_fn=lambda _idx: rp.retrieve_candidates(
                    query_vectors, k, query_texts=qtexts, mode=rmode
                ),
                item_fn=lambda i: rp.retrieve_candidates(
                    np.asarray(query_vectors[i : i + 1]),
                    k,
                    query_texts=_row_texts(i),
                    mode=rmode,
                )[0],
                node_id=node_id,
                placeholder=[],
                sink=s.drop_sink,
            )
            # Publish candidates as `retrieved` so a following rerank reads them (D2);
            # keyed so the rerank's finalize keeps the real query ids (M1d-2).
            _publish_retrieved(s, candidates, ids)
        else:
            _finalize_retrieval(
                s,
                isolate_batch(
                    ids,
                    list(range(nq)),
                    batch_fn=lambda _idx: rp.search_batch(
                        query_vectors, k, query_texts=qtexts
                    ),
                    item_fn=lambda i: rp.search_batch(
                        np.asarray(query_vectors[i : i + 1]),
                        k,
                        query_texts=_row_texts(i),
                    )[0],
                    node_id=node_id,
                    placeholder=[],
                    sink=s.drop_sink,
                ),
                query_vectors,
                ids,
                rp=rp,
            )
    s.cb("phase_3_retrieval", s.total, s.total, "Retrieval complete")


def _refine_inputs(s: RunState):
    """Shared inputs for the rerank / mmr / threshold refine nodes: the candidate
    ``retrieved`` (from the bound producer — retrieval or a prior refine node), the keyed
    ids, the indexed pipeline, and the per-query k (consistent — the retrieve node's k)."""
    candidates = s.get_artifact("retrieved")
    keyed = s.keyed_items("retrieved")
    ids = (
        [str(i) for i in keyed.ids]
        if keyed is not None
        else [str(i) for i in range(len(candidates))]
    )
    rp = s.get_artifact("vector_index", default=s.retrieval_pipeline)
    k = int(s.node_params.get("k", s.k))
    return candidates, ids, rp, k


@register_stage_handler("refine", time_key="retrieval_s")
def _stage_refine(s: RunState) -> None:
    """The ``refine`` operator: dispatch by op to rerank / mmr / threshold (each
    retrieved→retrieved; bodies unchanged)."""
    from ._dispatch import dispatch_operator

    dispatch_operator("refine", {
        "rerank": _stage_rerank,
        "mmr": _stage_mmr,
        "threshold": _stage_threshold,
    }, s)


def _stage_rerank(s: RunState) -> None:
    """Rerank node: reorder candidates (cross-encoder / token-overlap). Keeps the larger
    fetch_k pool when an MMR node follows (it re-selects k diverse from the pool), else
    truncates to k. Republishes ``retrieved`` for the next refine node / metrics."""
    s.cb("phase_3_5_rerank", 0, s.total, "Phase 3.5: Rerank")
    with TimingContext("Rerank Phase", logger):
        candidates, ids, rp, k = _refine_inputs(s)
        target = rp._fetch_k(k) if getattr(s, "mmr_in_graph", False) else k
        with _node_reranking(rp, s.node_params, getattr(s, "service_provider", None)):
            refined = rp.rerank_only(candidates, _retrieval_query_texts(s, rp), target)
        _publish_retrieved(s, refined, ids)
    s.cb("phase_3_5_rerank", s.total, s.total, "Rerank complete")


def _stage_mmr(s: RunState) -> None:
    """MMR node: re-select k diverse results per query (from the rerank pool / candidates)."""
    candidates, ids, rp, k = _refine_inputs(s)
    query_vectors = s.input("query_vectors", default=None)
    if isinstance(query_vectors, list) and len(query_vectors) > 0:
        query_vectors = np.array(query_vectors)
    _publish_retrieved(s, rp.mmr_only(candidates, query_vectors, k), ids)


def _stage_threshold(s: RunState) -> None:
    """Threshold node: truncate to k + drop results below the similarity threshold."""
    candidates, ids, rp, k = _refine_inputs(s)
    _publish_retrieved(s, rp.threshold_only(candidates, k), ids)


def _retrieved_from_bus(s: RunState):
    """The `retrieved` artifact from the bound producer (retrieval/rerank), as
    ``(results_with_scores, retrieved_keys, query_ids)`` locals (R4c, bus-only since
    M1c-2/M1d). Per-item identity rides the keyed ItemSet; index ids otherwise."""
    results_list = s.get_artifact("retrieved", default=[])
    keyed = s.keyed_items("retrieved")
    ids = (
        [str(i) for i in keyed.ids]
        if keyed is not None
        else [str(i) for i in range(len(results_list))]
    )
    return results_list, [_search_results_to_keys(r) for r in results_list], ids


def _stage_result_fusion(s: RunState) -> None:
    """Result-level fusion (second fusion level): combine the ranked results of the two
    upstream retrievals (audio-stream + text-stream) into one ranked set per query, reusing
    the registered hybrid rank-fusion strategies. Republishes ``retrieved`` so metrics /
    answer_gen bind the fused results."""
    from ..item_set import ItemSet
    from ...models.retrieval.fusion_registry import fuse_hybrid_results

    producers = s._producers("retrieved")
    sets = [
        s.ctx.get(pid, "retrieved")
        for pid in producers
        if s.ctx.has(pid, "retrieved")
    ]
    sets = [x for x in sets if isinstance(x, ItemSet)]
    if len(sets) < 2:
        # nothing to fuse — pass the single ranked set through unchanged.
        if sets:
            s.put_items("retrieved", sets[0])
        return

    params = s.node_params
    # Hybrid composition (R-hybrid): the two upstream arms are the dense + sparse retrievals
    # over ONE query stream. Fusion params come from the pipeline's core config (the same
    # hybrid_fusion_method / dense_weight / rrf_k the monolithic hybrid used), and the fused
    # pool keeps fetch_k depth when a refine node follows (so rerank/MMR see the full pool —
    # byte-identical to the in-pipeline hybrid path). The audio↔text result-level fusion keeps
    # its own params + per-query depth.
    hybrid = bool(params.get("hybrid"))
    rp = s.get_artifact("vector_index", default=s.retrieval_pipeline)
    core = getattr(getattr(rp, "strategy_config", None), "core", None)
    if hybrid and core is not None:
        method = params.get("method") or core.hybrid_fusion_method
        weight = float(params.get("weight", core.hybrid_dense_weight))
        rrf_k = int(params.get("rrf_k", core.rrf_k))
        k = int(params.get("k", s.k))
        hybrid_top_k = rp._fetch_k(k) if getattr(s, "refine_in_graph", False) else k
    else:
        method = params.get("method", "rrf")
        weight = float(params.get("weight", 0.5))
        rrf_k = int(params.get("rrf_k", 60))
        hybrid_top_k = None
    a, b = sets[0], sets[1]
    b_by_id = dict(zip(b.ids, b.values))
    fused_values = []
    for qid, a_results in zip(a.ids, a.values):
        b_results = b_by_id.get(qid, [])
        top_k = hybrid_top_k or int(
            params.get("top_k") or max(len(a_results), len(b_results)) or s.k
        )
        fused_values.append(
            fuse_hybrid_results(
                method,
                list(a_results),
                list(b_results),
                dense_weight=weight,
                top_k=top_k,
                rrf_k=rrf_k,
            )
        )
    s.put_items("retrieved", ItemSet(list(a.ids), fused_values))
    logger.info(
        "result_fusion: fused %d query result sets via %s", len(fused_values), method
    )
