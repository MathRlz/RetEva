"""Expressiveness sweep: every experiment shape the architecture promises (§15.3) is
authorable as an EXPLICIT graph.nodes config and builds with the right wiring.

The 46 repo configs cover the common shapes (golden lock); these tests pin the shapes no
config exercises yet — the composite / multi-instance / sink corners — so "all experiments
can be created in this manner" is a tested property, not a claim.
"""
import copy

from evaluator.config.evaluation import EvaluationConfig
from evaluator.config.graph_config import build_evaluation_config_kwargs
from evaluator.pipeline.graph.modes import build_graph_for_config

_DATASET = {
    "id": "pubmed_qa",
    "questions": "examples/data/pubmed_qa_small/questions.json",
    "corpus": "examples/data/pubmed_qa_small/corpus.json",
}


def _build(graph, **top):
    payload = copy.deepcopy({"dataset": _DATASET, "graph": graph, **top})
    # post-cut-over a config needs port-level edges: generate via the sanctioned authoring
    # tool (the corpus_merge case appends its ordering extras — and keeps today's
    # double-binding routing for parity; the deliberate fix is E7, TODO-pinned there)
    from evaluator.pipeline.graph.wiring import emit_edges

    g = payload["graph"]
    given = list(g.get("edges") or [])
    routed = {(e["to"], e["input"]) for e in given if "input" in e}
    g["edges"] = [
        e for e in emit_edges(g["nodes"]) if (e["to"], e["input"]) not in routed
    ] + given  # an explicitly-routed input REPLACES the generator's edges for that port
    cfg = EvaluationConfig.from_dict(build_evaluation_config_kwargs(payload), validate=False)
    return build_graph_for_config(cfg)


def _node(g, nid):
    return next(n for n in g.nodes if n.id == nid)


def _binds(g, nid):
    return set(_node(g, nid).bindings)


# ── RAG answer generation + scoring + judge (full chain) ─────────────────────────


def test_rag_generation_scoring_and_judge_chain():
    g = _build({"nodes": [
        "dataset_source", "corpus_embedding", "vector_db", "tts", "asr", "text_embedding",
        "retrieval",
        {"id": "answer_gen", "type": "answer_gen", "params": {"method": "simple"}},
        "answer_metrics", "retrieval_metrics", "metrics", "build_query_traces",
        "answer_judge", "finalize",
    ]})
    assert ("retrieved", "retrieval") in _binds(g, "answer_gen")          # grounded RAG
    assert ("short_answers", "dataset_source") in _binds(g, "answer_gen")  # answer GT rides in
    assert ("generated_answers", "answer_gen") in _binds(g, "answer_metrics")
    assert ("query_traces", "build_query_traces") in _binds(g, "answer_judge")


def test_closed_book_generation_no_retrieval():
    # `retrieved` is OPTIONAL context: no corpus/retrieval nodes → closed-book QA still builds,
    # generation reads the question straight from the source.
    g = _build({"nodes": [
        "dataset_source",
        {"id": "answer_gen", "type": "answer_gen", "params": {"method": "simple"}},
        "answer_metrics", "metrics", "finalize",
    ]})
    b = _binds(g, "answer_gen")
    assert ("query_text", "dataset_source") in b
    assert not any(art == "retrieved" for art, _ in b)                    # truly context-free


# ── fan-out retrieval composite (multi_query / decompose) ────────────────────────


def test_multi_query_retrieval_composite():
    g = _build({"nodes": [
        "dataset_source", "corpus_embedding", "vector_db", "tts", "asr",
        {"id": "multi_query_retrieval", "type": "multi_query_retrieval",
         "params": {"method": "decompose", "combine_strategy": "rrf"}},
        "retrieval_metrics", "metrics", "finalize",
    ]})
    mqr = _node(g, "multi_query_retrieval")
    assert dict(mqr.params).get("method") == "decompose"                  # variant survives load
    b = set(mqr.bindings)
    assert ("query_text", "asr") in b                                     # expands the hypothesis
    assert ("vector_index", "vector_db") in b
    assert ("retrieved", "multi_query_retrieval") in _binds(g, "retrieval_metrics")


# ── iterative RAG: hops unrolled as explicit instances ───────────────────────────


def test_iterative_rag_hops_explicit():
    g = _build({"nodes": [
        "dataset_source", "corpus_embedding", "vector_db", "tts", "asr",
        {"id": "text_embedding@h1", "type": "text_embedding"},
        {"id": "retrieval@h1", "type": "retrieval"},
        {"id": "query_refine@h1", "type": "query_refine",
         "params": {"method": "rewrite_with_context", "context_top_k": 2}},
        {"id": "text_embedding@h2", "type": "text_embedding"},
        {"id": "retrieval@h2", "type": "retrieval"},
        "retrieval_metrics", "metrics", "finalize",
    ]})
    # hop 2's embedder reads hop 1's refined query; hop 2's retrieval reads hop 2's vectors
    assert ("refined_query_text", "query_refine@h1") in _binds(g, "text_embedding@h2")
    assert ("text_query_vectors", "text_embedding@h2") in _binds(g, "retrieval@h2")
    assert ("retrieved", "retrieval@h1") in _binds(g, "query_refine@h1")


# ── refine chain: reorder + cascade + threshold ──────────────────────────────────


def test_refine_chain_reorder_cascade_threshold():
    g = _build({"nodes": [
        "dataset_source", "corpus_embedding", "vector_db", "tts", "asr", "text_embedding",
        "retrieval",
        "threshold",                                   # declared order wins: threshold FIRST
        "rerank",
        "mmr",
        {"id": "rerank@2", "type": "rerank"},          # cascade: a second rerank after mmr
        "retrieval_metrics", "metrics", "finalize",
    ]})
    assert ("retrieved", "retrieval") in _binds(g, "threshold")
    assert ("retrieved", "threshold") in _binds(g, "rerank")
    assert ("retrieved", "rerank") in _binds(g, "mmr")
    assert ("retrieved", "mmr") in _binds(g, "rerank@2")
    # metrics score the END of the chain (newest producer)
    rm = [b for b in _node(g, "retrieval_metrics").bindings if b[0] == "retrieved"]
    assert rm[-1] == ("retrieved", "rerank@2")


# ── multi-corpus: corpus_merge unions two embedded corpora into one index ────────


def test_corpus_merge_two_corpora_one_index():
    g = _build(
        {"nodes": [
            {"id": "src_a", "type": "dataset_source", "params": {"dataset": "a", "role": "corpus"}},
            {"id": "src_b", "type": "dataset_source", "params": {"dataset": "b", "role": "corpus"}},
            {"id": "qa", "type": "dataset_source",
             "params": {"dataset": "qa", "role": "questions"}},
            {"id": "ce_a", "type": "corpus_embedding"},
            {"id": "ce_b", "type": "corpus_embedding"},
            "corpus_merge", "vector_db", "tts", "asr", "text_embedding", "retrieval",
            "retrieval_metrics", "metrics", "finalize",
        ],
         # E7: port-level edges ROUTE each corpus to its own embedder — the thing
         # auto-wiring could never express (it bound BOTH sources to both embedders).
         "edges": [
             {"from": "src_a", "output": "corpus", "to": "ce_a", "input": "corpus"},
             {"from": "src_b", "output": "corpus", "to": "ce_b", "input": "corpus"},
         ]},
        datasets={
            "a": {"questions": _DATASET["questions"], "corpus": _DATASET["corpus"],
                  "id": "pubmed_qa", "role": "corpus"},
            "b": {"questions": _DATASET["questions"], "corpus": _DATASET["corpus"],
                  "id": "pubmed_qa", "role": "corpus"},
            "qa": {"questions": _DATASET["questions"], "corpus": _DATASET["corpus"],
                   "id": "pubmed_qa", "role": "questions"},
        },
    )
    merge = _binds(g, "corpus_merge")
    assert ("corpus_vectors", "ce_a") in merge and ("corpus_vectors", "ce_b") in merge
    # the deliberate E7 fix: each embedder reads ONLY its own source — auto-wiring bound both
    # sources to both embedders (they silently embedded the same corpus twice)
    assert _binds(g, "ce_a") == {("corpus", "src_a")}
    assert _binds(g, "ce_b") == {("corpus", "src_b")}
    # the index is built over the UNION (newest corpus_vectors producer = the merge)
    vdb = [b for b in _node(g, "vector_db").bindings if b[0] == "corpus_vectors"]
    assert vdb[-1] == ("corpus_vectors", "corpus_merge")


# ── query-set union: two question sources feed one pipeline ──────────────────────


def test_dataset_union_two_question_sources():
    g = _build(
        {"nodes": [
            {"id": "src_a", "type": "dataset_source",
             "params": {"dataset": "a", "role": "questions"}},
            {"id": "src_b", "type": "dataset_source",
             "params": {"dataset": "b", "role": "questions"}},
            "dataset_union",
            {"id": "corpus_src", "type": "dataset_source",
             "params": {"dataset": "docs", "role": "corpus"}},
            "corpus_embedding", "vector_db", "tts", "asr", "text_embedding", "retrieval",
            "retrieval_metrics", "metrics", "finalize",
        ]},
        datasets={
            "a": {"questions": _DATASET["questions"], "corpus": _DATASET["corpus"],
                  "id": "pubmed_qa", "role": "questions"},
            "b": {"questions": _DATASET["questions"], "corpus": _DATASET["corpus"],
                  "id": "pubmed_qa", "role": "questions"},
            "docs": {"questions": _DATASET["questions"], "corpus": _DATASET["corpus"],
                     "id": "pubmed_qa", "role": "corpus"},
        },
    )
    union = _binds(g, "dataset_union")
    assert ("query_text", "src_a") in union and ("query_text", "src_b") in union
    # downstream consumes the UNIONED stream (newest producer)
    tts = [b for b in _node(g, "tts").bindings if b[0] == "query_text"]
    assert tts[-1] == ("query_text", "dataset_union")


# ── report persistence: every sink is an explicit node ───────────────────────────


def test_all_three_report_sinks():
    g = _build({"nodes": [
        "dataset_source", "corpus_embedding", "vector_db", "tts", "asr", "text_embedding",
        "retrieval", "retrieval_metrics", "metrics", "finalize", "aggregate",
        "leaderboard_sink", "tracking_sink",
        {"id": "dataset_sink", "type": "dataset_sink", "params": {"output_dir": "out"}},
    ]})
    for sink in ("leaderboard_sink", "tracking_sink"):
        # the report bundle rides the bus as the `metrics` artifact (the report node's output)
        assert ("metrics", "metrics") in _binds(g, sink), sink


# ── precomputed-vector columns: embed-free retrieval (T4) ────────────────────────


def test_precomputed_vector_columns_embed_free_graph():
    g = _build({"nodes": [
        {"id": "dataset_source", "type": "dataset_source",
         "params": {"fields": {"qvec": "query_vectors", "cvec": "corpus_vectors",
                    "question": "query_text", "relevance": "relevant_docs"},
                    "embedding_space": "thirdparty_space"}},
        "vector_db", "retrieval", "retrieval_metrics", "metrics", "finalize",
    ]})
    assert ("corpus_vectors", "dataset_source") in _binds(g, "vector_db")
    assert ("query_vectors", "dataset_source") in _binds(g, "retrieval")


# ── cross-modal result-level fusion: two pinned retrieval arms ───────────────────


def test_cross_modal_result_fusion_pinned_arms():
    g = _build({"nodes": [
        "dataset_source", "corpus_embedding", "vector_db", "tts", "asr",
        "audio_embedding", "text_embedding",
        {"id": "retrieval_audio", "type": "retrieval",
         "params": {"vectors": "audio_query_vectors"}},
        {"id": "retrieval_text", "type": "retrieval",
         "params": {"vectors": "text_query_vectors"}},
        "result_fusion", "retrieval_metrics", "metrics", "finalize",
    ]})
    assert dict(_node(g, "retrieval_audio").params).get("vectors") == "audio_query_vectors"
    assert dict(_node(g, "retrieval_text").params).get("vectors") == "text_query_vectors"
    rf = _binds(g, "result_fusion")
    assert ("retrieved", "retrieval_audio") in rf and ("retrieved", "retrieval_text") in rf
