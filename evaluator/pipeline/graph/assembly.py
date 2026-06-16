"""Declarative graph assembly: feature-set → ordered node specs.

Replaces the former 14-parameter ``_mode_node_ids`` god-function + flag threading. The
execution DAG is assembled from a small :class:`FeatureSet` (one object instead of 14
positional flags) by placing each node at a **pipeline slot** and stable-sorting:

    source · index · query_head · query_prep · retrieve · refine · metrics · report ·
    generate · judge · finalize · sink

Adding an optional node is a ONE-LINE change here (`if f.x: items.append((SLOT, "node"))`)
plus a `FeatureSet` field — no signature threading. The genuinely-structural choices (which
embedder/retrieval variant generates candidates) live in the single ``_retrieve_core``
selector, not scattered across the assembler. Auto-wiring (`wiring._wire_nodes`) derives all
edges from the ordered list; slots only give the coarse order data-deps can't (e.g.
correction-before-optimization on the shared query-text chain).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Tuple

# Pipeline slots (ordinals). Floats leave room to interleave (retrieve-core sub-steps).
SLOT_SOURCE = 0.0
SLOT_TTS = 1.0
SLOT_INDEX = 2.0  # corpus_embedding (2.0) → vector_db (2.5)
SLOT_QUERY_HEAD = 4.0  # asr
SLOT_QUERY_PREP = 5.0  # correction (5.0) → optimization (6.0)
SLOT_RETRIEVE = 7.0  # the retrieve-core block (sub-ordinals 7.0, 7.01, …)
SLOT_REFINE = 8.0  # rerank
SLOT_TRANSCRIPTION_METRICS = 9.0
SLOT_RETRIEVAL_METRICS = 9.5
SLOT_REPORT = 10.0  # metrics (report assembler)
SLOT_GENERATE = 11.0  # answer_gen
SLOT_JUDGE = 12.0  # answer_judge
SLOT_FINALIZE = 13.0
SLOT_SINK = 14.0


@dataclass(frozen=True)
class FeatureSet:
    """The optional capabilities + structural choices that shape the graph. One object
    threaded in place of the old 14 positional flags; derived once from config (or from
    ``build_stage_graph`` kwargs in tests)."""

    embedding_fusion_enabled: bool = False
    result_fusion_enabled: bool = False
    query_opt_enabled: bool = False
    query_opt_method: str = "rewrite"
    hybrid_retrieval: bool = False
    rerank_enabled: bool = False
    mmr_enabled: bool = False
    threshold_enabled: bool = False
    tts_enabled: bool = False
    sink_enabled: bool = False
    correction_enabled: bool = False
    answer_gen_enabled: bool = False
    judge_enabled: bool = False
    trace_enabled: bool = False
    rag_rounds: int = 1
    refine_method: str = "rewrite_with_context"
    refine_context_top_k: int = 3

    @classmethod
    def maximal(cls) -> "FeatureSet":
        """Every optional node on — used to derive a mode's maximal required-model set."""
        return cls(
            embedding_fusion_enabled=True,
            query_opt_enabled=True,
            rerank_enabled=True,
            mmr_enabled=True,
            threshold_enabled=True,
            tts_enabled=True,
            sink_enabled=True,
            correction_enabled=True,
            answer_gen_enabled=True,
            judge_enabled=True,
            trace_enabled=True,
        )


def _embed_retrieve_hops(f: FeatureSet) -> List[Any]:
    """The embed→retrieve(→refine) hops for a text-query flow. ``rounds<=1`` is the plain
    single ``text_embedding``/``retrieval``; >1 unrolls into distinct instances with a
    ``query_refine`` between hops (the OneOf + prior-producer wiring binds each hop's
    embedder to the latest refined query and each retrieval to its own vectors)."""
    if f.rag_rounds <= 1:
        return ["text_embedding", "retrieval"]
    hops: List[Any] = []
    for i in range(1, f.rag_rounds + 1):
        hops.append({"id": f"text_embedding@h{i}", "type": "text_embedding"})
        hops.append({"id": f"retrieval@h{i}", "type": "retrieval"})
        if i < f.rag_rounds:
            hops.append(
                {
                    "id": f"query_refine@h{i}",
                    "type": "query_refine",
                    "params": {
                        "method": f.refine_method,
                        "context_top_k": f.refine_context_top_k,
                    },
                }
            )
    return hops


def _retrieve_core(mode: str, f: FeatureSet) -> List[Any]:
    """The ONE structural selector: how candidates are generated for a mode (the only place
    the mutually-exclusive embed/retrieve variants live)."""
    if mode == "asr_text_retrieval":
        if f.query_opt_enabled and f.query_opt_method in ("decompose", "multi_query"):
            # RAG-fusion composite REPLACES embed+retrieve (it is the retrieved producer).
            return [
                {
                    "id": "multi_query_retrieval",
                    "type": "multi_query_retrieval",
                    "params": {"method": f.query_opt_method},
                }
            ]
        if f.hybrid_retrieval and f.rag_rounds <= 1:
            # Hybrid as composition: one embedder feeds the dense arm; a sparse retrieval
            # over the query TEXT is the second arm; result_fusion combines the two ranked
            # sets (reusing the hybrid rank-fusion strategies) — no monolithic in-pipeline fuse.
            return [
                "text_embedding",
                {"id": "retrieval", "type": "retrieval", "params": {"mode": "dense"}},
                {"id": "retrieval_sparse", "type": "retrieval",
                 "params": {"mode": "sparse"}},
                {"id": "result_fusion", "type": "result_fusion",
                 "params": {"hybrid": True}},
            ]
        return _embed_retrieve_hops(f)
    if mode == "audio_emb_retrieval":
        return ["audio_embedding", "retrieval"]
    if mode == "audio_text_retrieval":
        if f.result_fusion_enabled:
            # Result level: retrieve the two streams separately, fuse the ranked results.
            return [
                "audio_embedding",
                "text_embedding",
                {"id": "retrieval_audio", "type": "retrieval",
                 "params": {"vectors": "audio_query_vectors"}},
                {"id": "retrieval_text", "type": "retrieval",
                 "params": {"vectors": "text_query_vectors"}},
                "result_fusion",
            ]
        if f.embedding_fusion_enabled:
            # Embedding level: fuse the vectors, then one retrieval.
            return ["audio_embedding", "text_embedding", "fusion", "retrieval"]
        return ["audio_embedding", "retrieval"]
    raise ValueError(f"Unknown pipeline mode: {mode}")


def assemble_specs(mode: str, f: FeatureSet) -> List[Any]:
    """The ordered node specs for a mode + feature set. Each line places one node at a slot;
    `_sorted` stable-sorts by slot. The result feeds ``build_graph_from_spec`` (auto-wiring).
    """
    items: List[Tuple[float, Any]] = [(SLOT_SOURCE, "dataset_source")]
    if f.tts_enabled:
        items.append((SLOT_TTS, "tts"))

    if mode == "asr_only":
        items += [
            (SLOT_QUERY_HEAD, "asr"),
            (SLOT_TRANSCRIPTION_METRICS, "transcription_metrics"),
            (SLOT_REPORT, "metrics"),
            (SLOT_FINALIZE, "finalize"),
        ]
        if f.sink_enabled:
            items.append((SLOT_SINK, "dataset_sink"))
        return _sorted(items)

    # retrieval modes: corpus index branch
    items += [(SLOT_INDEX, "corpus_embedding"), (SLOT_INDEX + 0.5, "vector_db")]

    # query head + prep (ASR text-query side only)
    if mode == "asr_text_retrieval":
        items.append((SLOT_QUERY_HEAD, "asr"))
        if f.correction_enabled:
            items.append((SLOT_QUERY_PREP, "query_correction"))
        # pre-retrieval rewrite/HyDE (the fan-out methods are the retrieve-core composite)
        if f.query_opt_enabled and f.query_opt_method in ("rewrite", "hyde"):
            items.append((SLOT_QUERY_PREP + 1.0, "query_optimization"))

    # the structural retrieve-core block (sub-ordinals preserve its internal order)
    for i, spec in enumerate(_retrieve_core(mode, f)):
        items.append((SLOT_RETRIEVE + i * 0.01, spec))

    # refine chain (each consumes+produces retrieved): rerank → mmr → threshold
    if f.rerank_enabled:
        items.append((SLOT_REFINE, "rerank"))
    if f.mmr_enabled:
        items.append((SLOT_REFINE + 0.1, "mmr"))
    if f.threshold_enabled:
        items.append((SLOT_REFINE + 0.2, "threshold"))
    # typed comparison nodes: transcription only when ASR ran; retrieval always (retrieving)
    if mode == "asr_text_retrieval":
        items.append((SLOT_TRANSCRIPTION_METRICS, "transcription_metrics"))
    # audio↔text alignment diagnostic: only when both vector streams exist (fusion modes)
    if mode == "audio_text_retrieval" and (
        f.embedding_fusion_enabled or f.result_fusion_enabled
    ):
        items.append((SLOT_TRANSCRIPTION_METRICS + 0.2, "embedding_alignment_metrics"))
    items.append((SLOT_RETRIEVAL_METRICS, "retrieval_metrics"))
    items.append((SLOT_REPORT, "metrics"))
    if f.answer_gen_enabled:
        items.append((SLOT_GENERATE, "answer_gen"))
        items.append((SLOT_GENERATE + 0.1, "answer_metrics"))
    # explicit trace builder: present when tracing is on or the judge (which reads traces) is
    if f.trace_enabled or f.judge_enabled:
        items.append((SLOT_GENERATE + 0.5, "build_query_traces"))
    if f.judge_enabled:
        items.append((SLOT_JUDGE, "answer_judge"))
    items.append((SLOT_FINALIZE, "finalize"))
    if f.sink_enabled:
        items.append((SLOT_SINK, "dataset_sink"))
    return _sorted(items)


def _sorted(items: List[Tuple[float, Any]]) -> List[Any]:
    """Stable-sort the (slot, spec) pairs by slot and return the specs."""
    return [spec for _slot, spec in sorted(items, key=lambda it: it[0])]
