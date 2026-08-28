"""Stage-graph topology per pipeline mode (the key node invariants)."""

from evaluator.pipeline import build_stage_graph
from evaluator.pipeline.graph.modes import label_from_graph


def _ids(mode, **kw):
    return set(build_stage_graph(mode, **kw).node_ids())


def test_label_from_graph_derives_mode_from_node_kinds():
    """The run/leaderboard label comes from the explicit graph's node kinds (no graph.mode):
    audio_embedding+text_embedding ⇒ audio_text; audio_embedding alone ⇒ audio_emb; asr(+retrieval)
    ⇒ asr_text/asr_only. Friendly names and canonical operator forms resolve the same."""
    L = lambda ns: label_from_graph({"nodes": ns})  # noqa: E731
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
    from evaluator.pipeline.graph.modes import _wire_mode_graph

    nodes = [
        "dataset_source",
        {"id": "asr_ref", "type": "asr", "params": {"oracle": True}},
        {"id": "asr_raw", "type": "asr"},
    ]
    edges = [
        {"from": "dataset_source", "to": "asr_ref", "input": "query_audio"},
        {"from": "dataset_source", "to": "asr_raw", "input": "query_audio"},
    ]
    graph = _wire_mode_graph(
        None, None, graph_override={"nodes": nodes, "edges": edges},
        config=None, attach_fields=False,
    )
    ids = [n.id for n in graph.nodes]
    assert ids.count("dataset_source") == 1  # naturally shared, no duplication
    assert "asr_ref" in ids and "asr_raw" in ids


def test_asr_text_retrieval_topology():
    ids = _ids("asr_text_retrieval")
    assert {"dataset_source", "asr", "text_embedding", "retrieval"} <= ids
    assert "audio_embedding" not in ids


def test_audio_emb_retrieval_topology():
    ids = _ids("audio_emb_retrieval")
    assert {"dataset_source", "audio_embedding", "corpus_embedding", "retrieval"} <= ids
    assert "asr" not in ids
    assert "text_embedding" not in ids  # corpus_embedding handles the text corpus


def test_audio_text_retrieval_has_fusion():
    ids = _ids("audio_text_retrieval", embedding_fusion_enabled=True)
    assert {"audio_embedding", "text_embedding", "fusion"} <= ids


def test_asr_only_has_no_retrieval():
    ids = _ids("asr_only")
    assert "asr" in ids
    assert "retrieval" not in ids
    assert "vector_db" not in ids


def test_rerank_inserted_when_enabled():
    base = _ids("asr_text_retrieval")
    with_rerank = _ids("asr_text_retrieval", rerank_enabled=True)
    assert "rerank" not in base
    assert "rerank" in with_rerank


# ── Composable refine chain (Roadmap 2a) ─────────────────────────────


def _refine_chain_ids(mode, **kw):
    """Refine node ids in execution order for a built graph."""
    g = build_stage_graph(mode, **kw)
    return [n.id for n in g.nodes if n.params.get("op") in ("rerank", "mmr", "threshold")]


def test_refine_chain_default_order_from_flags():
    # No refine_ops → canonical rerank → mmr → threshold from the enabled flags.
    chain = _refine_chain_ids(
        "asr_text_retrieval",
        rerank_enabled=True, mmr_enabled=True, threshold_enabled=True,
    )
    assert chain == ["rerank", "mmr", "threshold"]


def test_refine_ops_reorders_the_chain():
    # Explicit refine_ops wins verbatim, letting a researcher reorder.
    chain = _refine_chain_ids("asr_text_retrieval", refine_ops=("threshold", "rerank", "mmr"))
    assert chain == ["threshold", "rerank", "mmr"]


def test_refine_ops_repeat_cascades_with_unique_ids():
    # A repeated op cascades (each binds the prior producer); ids stay unique.
    g = build_stage_graph("asr_text_retrieval", refine_ops=("rerank", "mmr", "rerank"))
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
