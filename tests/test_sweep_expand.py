"""Axes-driven graph expansion (cli/sweep_expand.py) — the no-graph.branches sweep tool."""

import pytest

from evaluator.cli.sweep_expand import (
    SweepExpandError,
    _affected_set,
    _combo_label,
    _forward_adjacency,
    expand_axes,
    main,
)

BASE_NODES = [
    "dataset_source",
    "corpus_embedding",
    "vector_db",
    "text_embedding",
    "retrieval",
    {"id": "metrics", "type": "metrics"},
]

BASE_EDGES = [
    {"from": "dataset_source", "to": "corpus_embedding", "input": "corpus"},
    {"from": "corpus_embedding", "to": "vector_db", "input": "corpus_vectors"},
    {"from": "dataset_source", "to": "text_embedding", "input": "query_text"},
    {"from": "text_embedding", "to": "retrieval", "input": "query_vectors"},
    {"from": "vector_db", "to": "retrieval", "input": "vector_index"},
    {"from": "retrieval", "to": "metrics", "input": "retrieved"},
]


def _axes(*axis_dicts):
    return {"axes": list(axis_dicts)}


def test_forward_adjacency_and_affected_set():
    adj = _forward_adjacency(
        ["a", "b", "c"], [{"from": "a", "to": "b"}, {"from": "b", "to": "c"}]
    )
    assert adj == {"a": {"b"}, "b": {"c"}, "c": set()}
    assert _affected_set({"b"}, adj) == {"b", "c"}
    assert _affected_set({"a"}, adj) == {"a", "b", "c"}


def test_expand_single_axis_shares_upstream_nodes():
    axes = _axes({"node": "text_embedding", "param": "model", "values": ["jina_v4", "labse"]})
    new_nodes, new_edges, combos = expand_axes(BASE_NODES, BASE_EDGES, axes)

    assert [c["label"] for c in combos] == ["jina_v4", "labse"]

    ids = [n if isinstance(n, str) else n["id"] for n in new_nodes]
    # Upstream of text_embedding (dataset_source) stays a single shared node.
    assert ids.count("dataset_source") == 1
    assert ids.count("corpus_embedding") == 1
    assert ids.count("vector_db") == 1
    # text_embedding is the varied node itself, downstream (retrieval, metrics) follow it.
    assert "text_embedding_jina_v4" in ids
    assert "text_embedding_labse" in ids
    assert "retrieval_jina_v4" in ids
    assert "metrics_labse" in ids
    assert "text_embedding" not in ids  # the un-suffixed original is gone

    # Each variant's text_embedding node carries its own model param.
    by_id = {n["id"]: n for n in new_nodes if isinstance(n, dict)}
    assert by_id["text_embedding_jina_v4"]["params"] == {"model": "jina_v4"}
    assert by_id["text_embedding_labse"]["params"] == {"model": "labse"}

    # dataset_source -> corpus_embedding is untouched by the sweep, kept once.
    shared = [e for e in new_edges if e["from"] == "dataset_source" and e["to"] == "corpus_embedding"]
    assert len(shared) == 1
    # dataset_source -> text_embedding fans out to both variants.
    fanout = [e for e in new_edges if e["from"] == "dataset_source" and e["to"].startswith("text_embedding_")]
    assert {e["to"] for e in fanout} == {"text_embedding_jina_v4", "text_embedding_labse"}


def test_expand_multi_axis_combines_overrides_per_node():
    axes = _axes(
        {"node": "text_embedding", "param": "model", "values": ["jina_v4", "labse"]},
        {"node": "retrieval", "param": "k", "values": [5, 10]},
    )
    new_nodes, _new_edges, combos = expand_axes(BASE_NODES, BASE_EDGES, axes)
    assert len(combos) == 4
    assert {c["label"] for c in combos} == {"jina_v4_5", "jina_v4_10", "labse_5", "labse_10"}

    by_id = {n["id"]: n for n in new_nodes if isinstance(n, dict)}
    assert by_id["retrieval_jina_v4_5"]["params"] == {"k": 5}
    assert by_id["text_embedding_jina_v4_5"]["params"] == {"model": "jina_v4"}


def test_dotted_param_path_deep_merges_into_existing_params():
    nodes = list(BASE_NODES[:-2]) + [
        {"id": "retrieval", "type": "retrieval", "params": {"k": 10, "fusion": {"method": "linear"}}},
    ]
    axes = _axes({"node": "retrieval", "param": "fusion.method", "values": ["rrf"]})
    new_nodes, _e, _c = expand_axes(nodes, BASE_EDGES, axes)
    by_id = {n["id"]: n for n in new_nodes if isinstance(n, dict)}
    assert by_id["retrieval_rrf"]["params"] == {"k": 10, "fusion": {"method": "rrf"}}


def test_unknown_axis_node_raises():
    axes = _axes({"node": "nope", "param": "model", "values": ["a"]})
    with pytest.raises(SweepExpandError, match="not in base graph.nodes"):
        expand_axes(BASE_NODES, BASE_EDGES, axes)


def test_colliding_labels_raise():
    axes = _axes({"node": "retrieval", "param": "k", "values": ["5", 5]})
    with pytest.raises(SweepExpandError, match="same id suffix"):
        expand_axes(BASE_NODES, BASE_EDGES, axes)


def test_combo_label_joins_sanitized_values():
    assert _combo_label([("retrieval", "mode", "dense/linear")]) == "dense-linear"
    assert _combo_label([("a", "x", "v1"), ("b", "y", 5)]) == "v1_5"


def test_expand_axes_requires_explicit_edges(tmp_path):
    """A base config with graph.nodes but no port-level graph.edges is refused: the expansion
    rewires by ``from``/``to``, so an implicitly chained graph has nothing to rewire."""
    import json

    base = tmp_path / "base.yaml"
    base.write_text(json.dumps({"graph": {"nodes": BASE_NODES}}), encoding="utf-8")
    axes = tmp_path / "axes.json"
    axes.write_text(
        json.dumps(_axes({"node": "retrieval", "param": "k", "values": [5, 10]})),
        encoding="utf-8",
    )

    with pytest.raises(SweepExpandError, match="no explicit graph.edges"):
        main(["--base", str(base), "--axes", str(axes), "--dry-run"])
