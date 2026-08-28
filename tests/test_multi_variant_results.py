"""R4: multi-variant graphs (N `metrics`/`finalize` node pairs sharing an upstream producer)
must not lose every variant but one. `RunState.results` is a single shared dict — before the
R4 fix, `_stage_metrics`'s `s.results = results` and `_stage_finalize`'s in-place mutation
meant only the LAST-executed metrics/finalize node's numbers survived. This pins the fix at
both levels: the low-level aggregation (`_collect_final_result`) and a full graph run.
"""

from evaluator.evaluation.executor.run import _collect_final_result
from evaluator.evaluation.run_context import RunContext


class _Node:
    def __init__(self, node_id, stage):
        self.id = node_id
        self.stage = stage
        self.params = {}


class _Graph:
    def __init__(self, nodes):
        self.nodes = nodes


class _State:
    def __init__(self, ctx, stage_times=None, node_times=None):
        self.ctx = ctx
        self.stage_times = stage_times or {}
        self.node_times = node_times or {}
        self.results = {"leftover": "shared-slot-should-not-be-used"}


def test_collect_final_result_wraps_multiple_finalize_reports_by_node_id():
    ctx = RunContext()
    ctx.put("finalize_a", "run_report", {"MRR": 0.9, "report": {}})
    ctx.put("finalize_b", "run_report", {"MRR": 0.2, "report": {}})
    graph = _Graph([_Node("finalize_a", "finalize"), _Node("finalize_b", "finalize")])
    state = _State(ctx)

    result = _collect_final_result(state, graph)

    assert set(result["variants"]) == {"finalize_a", "finalize_b"}
    assert result["variants"]["finalize_a"]["MRR"] == 0.9
    assert result["variants"]["finalize_b"]["MRR"] == 0.2
    assert "MRR" not in result  # not flattened/collapsed — each variant stays distinct


def test_collect_final_result_stays_flat_for_a_single_finalize_node():
    ctx = RunContext()
    ctx.put("finalize", "run_report", {"MRR": 0.5, "report": {}})
    graph = _Graph([_Node("finalize", "finalize")])
    state = _State(ctx)

    result = _collect_final_result(state, graph)

    assert result["MRR"] == 0.5
    assert "variants" not in result


def test_collect_final_result_falls_back_to_shared_results_without_a_finalize_node():
    graph = _Graph([_Node("some_other_node", "metrics")])
    state = _State(RunContext())

    result = _collect_final_result(state, graph)

    assert result == {"leftover": "shared-slot-should-not-be-used"}


# ── Full-graph run: two retrieval variants (different k) sharing one embedder/index ──
# Reuses `_run` from test_integration_audio_emb.py (same mock pipelines, same config
# defaults) so this test doesn't have to independently rediscover a working config shape.


def _variant_edges(variant: str) -> list:
    """The retrieval→retrieval_metrics→metrics→finalize tail for one variant, node-id
    suffixed, mirroring exactly what `emit_edges` derives for a single-chain graph (see
    test_integration_audio_emb.py) — hand-written because auto-wiring a SHARED-prefix,
    DIVERGENT-tail graph would ambiguously bind both variants' `retrieved` into each
    other's consumers (there being two producers of the same plain-port artifact name)."""
    r = f"retrieval_{variant}"
    rm = f"retrieval_metrics_{variant}"
    m = f"metrics_{variant}"
    f = f"finalize_{variant}"
    return [
        {"from": "audio_embedding", "output": "audio_query_vectors", "to": r, "input": "query_vectors"},
        {"from": "vector_db", "to": r, "input": "vector_index"},
        {"from": "dataset_source", "to": r, "input": "reference_transcription"},
        {"from": "dataset_source", "to": r, "input": "relevant_docs"},
        {"from": r, "to": rm, "input": "retrieved"},
        {"from": "dataset_source", "to": rm, "input": "relevant_docs"},
        {"from": rm, "to": m, "input": "retrieval_scores"},
        {"from": r, "to": m, "input": "retrieved"},
        {"from": "dataset_source", "to": m, "input": "reference_transcription"},
        {"from": "dataset_source", "to": m, "input": "relevant_docs"},
        {"from": rm, "to": m, "input": "per_query_recall5"},
        {"from": m, "to": f, "input": "metrics"},
        {"from": r, "to": f, "input": "retrieved"},
    ]


def test_two_retrieval_variants_both_survive_a_real_graph_run():
    from evaluator.config.graph_config import build_evaluation_config_kwargs

    from tests.test_integration_audio_emb import _run

    nodes = [
        "dataset_source",
        {"id": "audio_embedding", "type": "audio_embedding",
         "params": {"model": "attention_pool", "dim": 8}},
        "corpus_embedding",
        {"id": "vector_db", "type": "vector_db", "params": {"store": "inmemory"}},
        {"id": "retrieval_a", "type": "retrieval", "params": {"k": 1}},
        {"id": "retrieval_b", "type": "retrieval", "params": {"k": 8}},
        {"id": "retrieval_metrics_a", "type": "retrieval_metrics"},
        {"id": "retrieval_metrics_b", "type": "retrieval_metrics"},
        {"id": "metrics_a", "type": "metrics"},
        {"id": "metrics_b", "type": "metrics"},
        {"id": "finalize_a", "type": "finalize"},
        {"id": "finalize_b", "type": "finalize"},
    ]
    shared_prefix_edges = [
        {"from": "dataset_source", "to": "audio_embedding", "input": "query_audio"},
        {"from": "dataset_source", "to": "corpus_embedding", "input": "corpus"},
        {"from": "corpus_embedding", "to": "vector_db", "input": "corpus_vectors"},
    ]
    edges = shared_prefix_edges + _variant_edges("a") + _variant_edges("b")
    legacy = build_evaluation_config_kwargs(
        {"graph": {"nodes": nodes, "edges": edges}}
    )

    res = _run(graph_override=legacy["graph_override"])

    assert set(res["variants"]) == {"finalize_a", "finalize_b"}
    a, b = res["variants"]["finalize_a"], res["variants"]["finalize_b"]
    # Genuinely two independent report objects, not the same shared dict read twice (the R4
    # bug: before the fix, `s.results = results` in `_stage_metrics` meant only the LAST of
    # metrics_a/metrics_b to execute would leave any trace at all — `res` wouldn't even have a
    # "variants" key, let alone two populated ones).
    assert a is not b
    assert a.get("MRR") is not None and b.get("MRR") is not None
    assert a.get("Recall@5") is not None and b.get("Recall@5") is not None
