"""Same-name port edges may omit `output`: `{from, to, input: X}` means `output == X`.

1718 of the repo's 1810 port edges name the same artifact twice; only a OneOf rename
(`text_query_vectors` → `query_vectors`) needs both fields. The shorthand must build the
IDENTICAL graph as the long form — it is spelling, not semantics.
"""
import pytest

from evaluator.config.graph_config import GraphConfigError, build_evaluation_config_kwargs
from evaluator.pipeline.graph.wiring import (
    _wire_nodes,
    bind_explicit_edges,
    emit_edges,
    is_port_edge,
    normalize_port_edge,
)

from tests.test_explicit_edges import _EDGES, _NODES


def _shorten(edge):
    """Drop a redundant `output` — what a hand-authored config now looks like."""
    return ({k: v for k, v in edge.items() if k != "output"}
            if edge.get("output") == edge.get("input") else edge)


def test_shorthand_builds_the_same_graph_as_the_long_form():
    long_form = bind_explicit_edges(_NODES, _EDGES)
    short_form = bind_explicit_edges(_NODES, [_shorten(e) for e in _EDGES])
    for a, b in zip(long_form, short_form):
        assert a.id == b.id and a.stage == b.stage
        assert a.bindings == b.bindings, f"{a.id}: {a.bindings} != {b.bindings}"  # ORDER too
        assert a.input_aliases == b.input_aliases, a.id
        assert a.depends_on == b.depends_on, a.id


def test_shorthand_still_reproduces_the_autowired_graph():
    auto = _wire_nodes(_NODES)
    short = bind_explicit_edges(_NODES, [_shorten(e) for e in _EDGES])
    assert [n.bindings for n in auto] == [n.bindings for n in short]


def test_emitter_shortens_same_name_edges_and_keeps_renames():
    emitted = emit_edges(_NODES)
    assert emitted, "emitter produced nothing"
    for e in emitted:
        if "output" in e:
            assert e["output"] != e["input"], f"redundant output on {e}"
    renames = [e for e in emitted if "output" in e]
    assert {(e["output"], e["input"]) for e in renames} == {
        ("text_query_vectors", "query_vectors")
    }


def test_emitted_edges_round_trip():
    # The E3 contract: bind_explicit_edges(nodes, emit_edges(nodes)) == _wire_nodes(nodes).
    auto = _wire_nodes(_NODES)
    rebound = bind_explicit_edges(_NODES, emit_edges(_NODES))
    for a, b in zip(auto, rebound):
        assert a.bindings == b.bindings, a.id
        assert a.depends_on == b.depends_on, a.id


def test_output_without_input_is_still_an_error():
    with pytest.raises(ValueError, match="needs 'input'"):
        normalize_port_edge({"from": "a", "output": "x", "to": "b"})


def test_ordering_only_edge_is_not_a_port_edge():
    assert not is_port_edge({"from": "a", "to": "b"})
    assert is_port_edge({"from": "a", "to": "b", "input": "x"})


def test_loader_accepts_shorthand_and_rejects_output_only():
    graph = {"nodes": ["dataset_source", "corpus_embedding"],
             "edges": [{"from": "dataset_source", "to": "corpus_embedding", "input": "corpus"}]}
    legacy = build_evaluation_config_kwargs({"graph": graph, "experiment": {"name": "t"}})
    # normalized at load: downstream only ever sees the long form
    assert legacy["graph_override"]["edges"] == [
        {"from": "dataset_source", "to": "corpus_embedding",
         "input": "corpus", "output": "corpus"}
    ]

    graph["edges"] = [{"from": "dataset_source", "output": "corpus", "to": "corpus_embedding"}]
    with pytest.raises(GraphConfigError, match="needs 'input'"):
        build_evaluation_config_kwargs({"graph": graph, "experiment": {"name": "t"}})
