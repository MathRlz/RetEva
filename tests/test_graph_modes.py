"""Run-label derivation + the wiring shapes an explicit node list produces."""

from evaluator.pipeline.graph.build import graph_label
from evaluator.pipeline.graph.wiring import build_graph_from_spec, emit_edges


def _graph(nodes, edges=None):
    return build_graph_from_spec(nodes, mode="custom", edges=edges or emit_edges(nodes))


def test_graph_label_derives_the_run_label_from_node_kinds():
    """The run/leaderboard label comes from the explicit graph's node kinds (no graph.mode):
    audio_embedding+text_embedding ⇒ audio_text; audio_embedding alone ⇒ audio_emb; asr(+retrieval)
    ⇒ asr_text/asr_only. Friendly names and canonical operator forms resolve the same."""
    L = lambda ns: graph_label({"nodes": ns})  # noqa: E731
    assert L(["dataset_source", "asr", "text_embedding", "retrieval"]) == "asr_text_retrieval"
    assert L(["dataset_source", "asr"]) == "asr_only"
    assert L(["dataset_source", "audio_embedding", "retrieval"]) == "audio_emb_retrieval"
    assert L(["dataset_source", "audio_embedding", "text_embedding", "fusion",
              "retrieval"]) == "audio_text_retrieval"
    # canonical operator form resolves identically
    assert L([{"type": "convert", "params": {"op": "asr"}}, "retrieval",
              {"type": "embed", "params": {"axis": "query", "modality": "text"}}]) \
        == "asr_text_retrieval"
    assert L(["dataset_source", "corpus_embedding"]) is None


def test_comparing_variants_is_just_distinct_nodes_sharing_a_producer():
    """No graph.branches: comparing two ASR model variants is two distinctly-named `asr`
    nodes in the ONE node list, both wired from the same shared dataset_source via ordinary
    edges — the graph engine already supports this natively, nothing to expand or collapse."""
    nodes = [
        "dataset_source",
        {"id": "asr_ref", "type": "asr", "params": {"oracle": True}},
        {"id": "asr_raw", "type": "asr"},
    ]
    edges = [
        {"from": "dataset_source", "to": "asr_ref", "input": "query_audio"},
        {"from": "dataset_source", "to": "asr_raw", "input": "query_audio"},
    ]
    ids = [n.id for n in _graph(nodes, edges).nodes]
    assert ids.count("dataset_source") == 1  # naturally shared, no duplication
    assert "asr_ref" in ids and "asr_raw" in ids


def test_repeated_refine_op_cascades_with_unique_ids():
    # A repeated op cascades (each binds the prior producer); ids stay unique.
    nodes = [
        "dataset_source", "corpus_embedding", "vector_db", "asr", "text_embedding", "retrieval",
        "rerank", "mmr", {"id": "rerank@2", "type": "rerank"},
    ]
    g = _graph(nodes)
    chain = [n.id for n in g.nodes if n.params.get("op") in ("rerank", "mmr", "threshold")]
    assert chain == ["rerank", "mmr", "rerank@2"]
    # the second rerank consumes the mmr output, not the candidates
    second = next(n for n in g.nodes if n.id == "rerank@2")
    assert dict(second.bindings)["retrieved"] == "mmr"


def test_refine_ops_rejects_unknown_op():
    import pytest

    from evaluator.config.vector_db import VectorDBConfig

    with pytest.raises(ValueError, match="refine_ops"):
        VectorDBConfig(refine_ops=["rerank", "bogus"])
