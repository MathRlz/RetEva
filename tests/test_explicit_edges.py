"""Explicit port-level wiring (E2, docs/archive/ARCHITECTURE_TASKS.md).

`bind_explicit_edges` reconstructs bindings/aliases/deps from `{from, output, to, input}`
edges; a correct edge set reproduces the auto-wired graph BIT-for-bit (order included —
runtime reads are newest-producer-first + OneOf priority). Portless `{from, to}` entries add
ordering only. Validation errors name the offending edge in config vocabulary.
"""
import pytest

from evaluator.pipeline.graph.wiring import (
    _wire_nodes,
    bind_explicit_edges,
    build_graph_from_spec,
)

_NODES = [
    "dataset_source", "tts", "corpus_embedding", "vector_db", "asr",
    "text_embedding", "retrieval",
]

# the full port-level edge set for the TTS-bridge retrieval graph (run shape)
_EDGES = [
    {"from": "dataset_source", "output": "query_text", "to": "tts", "input": "query_text"},
    {"from": "dataset_source", "output": "corpus", "to": "corpus_embedding", "input": "corpus"},
    {"from": "corpus_embedding", "output": "corpus_vectors",
     "to": "vector_db", "input": "corpus_vectors"},
    {"from": "dataset_source", "output": "query_audio", "to": "asr", "input": "query_audio"},
    {"from": "tts", "output": "query_audio", "to": "asr", "input": "query_audio"},
    {"from": "dataset_source", "output": "reference_transcription",
     "to": "asr", "input": "reference_transcription"},
    {"from": "dataset_source", "output": "query_text",
     "to": "text_embedding", "input": "query_text"},
    {"from": "asr", "output": "query_text", "to": "text_embedding", "input": "query_text"},
    {"from": "text_embedding", "output": "text_query_vectors",
     "to": "retrieval", "input": "query_vectors"},
    {"from": "vector_db", "output": "vector_index", "to": "retrieval", "input": "vector_index"},
    {"from": "dataset_source", "output": "query_text", "to": "retrieval", "input": "query_text"},
    {"from": "asr", "output": "query_text", "to": "retrieval", "input": "query_text"},
    {"from": "dataset_source", "output": "reference_transcription",
     "to": "retrieval", "input": "reference_transcription"},
    {"from": "dataset_source", "output": "relevant_docs",
     "to": "retrieval", "input": "relevant_docs"},
]


def test_explicit_edges_reproduce_autowired_graph_bit_for_bit():
    auto = _wire_nodes(_NODES)
    explicit = bind_explicit_edges(_NODES, _EDGES)
    for a, e in zip(auto, explicit):
        assert a.id == e.id and a.stage == e.stage
        assert a.bindings == e.bindings, f"{a.id}: {a.bindings} != {e.bindings}"  # ORDER too
        assert a.input_aliases == e.input_aliases, a.id
        assert a.depends_on == e.depends_on, a.id


def test_yaml_edge_order_is_not_semantic():
    shuffled = list(reversed(_EDGES))
    auto = _wire_nodes(_NODES)
    explicit = bind_explicit_edges(_NODES, shuffled)
    assert [n.bindings for n in auto] == [n.bindings for n in explicit]


def test_edges_can_route_where_autowiring_cannot():
    # The point of the redesign: drop the dataset_source→text_embedding fallback — only asr
    # feeds the embedder. Auto-wiring can never express this.
    pruned = [e for e in _EDGES
              if not (e["from"] == "dataset_source" and e["to"] == "text_embedding")]
    g = bind_explicit_edges(_NODES, pruned)
    te = next(n for n in g if n.id == "text_embedding")
    assert te.bindings == (("query_text", "asr"),)


def test_ordering_only_edge_adds_dep_not_binding():
    edges = _EDGES + [{"from": "vector_db", "to": "text_embedding"}]
    g = bind_explicit_edges(_NODES, edges)
    te = next(n for n in g if n.id == "text_embedding")
    assert "vector_db" in te.depends_on
    assert all(prod != "vector_db" for _, prod in te.bindings)


def test_build_graph_from_spec_dispatches_on_edge_shape():
    explicit = build_graph_from_spec(_NODES, edges=_EDGES)
    auto = build_graph_from_spec(_NODES)
    assert [n.bindings for n in explicit.nodes] == [n.bindings for n in auto.nodes]


@pytest.mark.parametrize("bad, match", [
    ({"from": "nope", "output": "query_text", "to": "tts", "input": "query_text"},
     "unknown node 'nope'"),
    ({"from": "dataset_source", "output": "query_text", "to": "tts"},
     "needs 'input'"),
    ({"from": "dataset_source", "output": "not_an_artifact", "to": "tts",
      "input": "query_text"}, "does not produce"),
    ({"from": "dataset_source", "output": "query_text", "to": "tts", "input": "bogus_port"},
     "no input port 'bogus_port'"),
    ({"from": "dataset_source", "output": "corpus", "to": "tts", "input": "query_text"},
     "accepts"),
    ({"from": "retrieval", "output": "retrieved", "to": "asr", "input": "query_audio"},
     "must come before"),
])
def test_bad_edges_fail_with_named_errors(bad, match):
    with pytest.raises(ValueError, match=match):
        bind_explicit_edges(_NODES, _EDGES + [bad])


def test_duplicate_edge_rejected():
    with pytest.raises(ValueError, match="duplicate edge"):
        bind_explicit_edges(_NODES, _EDGES + [_EDGES[0]])


def test_duplicate_node_id_rejected():
    """No caller (builder rename, CLI YAML, an imported config) may silently collapse two nodes
    into one id — `index_of`'s dict comprehension would otherwise pick the last one and drop
    the other's params/edges with no error."""
    with pytest.raises(ValueError, match="duplicate node id 'retrieval'"):
        build_graph_from_spec(_NODES + ["retrieval"], edges=_EDGES)


def test_duplicate_node_id_rejected_across_string_and_dict_spec_shapes():
    # a bare string and a {id, type} dict for the SAME id both normalize to that id
    # (_normalize_spec_item) -- the check must catch the collision regardless of shape.
    nodes = _NODES + [{"id": "retrieval", "type": "retrieval"}]
    with pytest.raises(ValueError, match="duplicate node id 'retrieval'"):
        build_graph_from_spec(nodes, edges=_EDGES)


def test_missing_required_edge_names_the_fix():
    # drop retrieval's vector_index edge: the graph produces it, so its absence is an error
    pruned = [e for e in _EDGES if not (e["to"] == "retrieval" and e["input"] == "vector_index")]
    with pytest.raises(ValueError, match="required input 'vector_index' has no edge"):
        bind_explicit_edges(_NODES, pruned)


def test_branches_remap_port_edges_per_branch():
    from evaluator.pipeline.graph.branches import expand_branches

    specs, edges = expand_branches(
        _NODES, [{"id": "b1", "asr": {"oracle": True}}], edges=_EDGES)
    assert {"from": "tts@b1", "output": "query_audio",
            "to": "asr@b1", "input": "query_audio"} in edges
    g = build_graph_from_spec(specs, edges=edges)
    te = next(n for n in g.nodes if n.id == "text_embedding@b1")
    assert ("query_text", "asr@b1") in te.bindings


def test_emit_edges_reproduces_every_config_graph_bit_for_bit():
    """E3, the real parity proof: for EVERY repo config, reconstructing from the generated
    edge set reproduces the auto-wired RUN-shape graph as tuple equality — bindings including
    ORDER, OneOf input_aliases, and depends_on. Stronger than the sorted golden."""
    import glob
    import os

    import yaml

    from evaluator.config.evaluation import EvaluationConfig
    from evaluator.config.graph_config import build_evaluation_config_kwargs
    from evaluator.pipeline.graph.wiring import emit_edges

    root = os.path.dirname(os.path.dirname(__file__))
    configs = sorted(
        glob.glob(os.path.join(root, "configs/*.yaml"))
        + glob.glob(os.path.join(root, "configs/apm_tests/**/*.yaml"), recursive=True)
        + glob.glob(os.path.join(root, "configs/examples/*.yaml"))
        + glob.glob(os.path.join(root, "configs/campaign/*.yaml"))
    )
    assert len(configs) >= 46
    # Non-canonical topologies exist BECAUSE auto-wiring cannot express them (B2): the
    # inverted chain's hand-written edges intentionally diverge from the auto-wired graph
    # (chain-order ranking would bypass the corrector), so the "reconstruct == autowire"
    # property is meaningless there. Their explicit-edge round-trip is pinned instead in
    # test_typed_port_routing.py::test_routed_graph_edge_round_trip.
    non_canonical = {"showcase_correction_after_optimization.yaml"}
    for path in configs:
        name = os.path.basename(path)
        if name in non_canonical:
            continue
        raw = yaml.safe_load(open(path))
        cfg = EvaluationConfig.from_dict(build_evaluation_config_kwargs(raw), validate=False)
        nodes = (cfg.graph_override or {}).get("nodes") or []
        auto = _wire_nodes(nodes)                      # the RUN shape (no attach_fields)
        explicit = bind_explicit_edges(nodes, emit_edges(nodes))
        for a, e in zip(auto, explicit):
            assert a.bindings == e.bindings, f"{name}:{a.id} bindings"
            assert a.input_aliases == e.input_aliases, f"{name}:{a.id} aliases"
            assert a.depends_on == e.depends_on, f"{name}:{a.id} deps"


def test_migrated_configs_build_one_graph_for_preview_and_run():
    """E4: with explicit port-level edges the preview and run builders produce ONE graph —
    same bindings/aliases/deps everywhere; the preview only adds display params (fields/
    suppress_outputs on dataset_source), never wiring."""
    import os

    import yaml

    from evaluator.config.evaluation import EvaluationConfig
    from evaluator.config.graph_config import build_evaluation_config_kwargs
    from evaluator.pipeline.graph.modes import build_graph_for_config
    from tests.graph_golden_util import build_run_shape

    root = os.path.dirname(os.path.dirname(__file__))
    for name in ("e2e_pubmed_qa_small", "showcase_hybrid_rerank_mmr",
                 "evaluation_config_admed_selfretr_fusion", "e2e_pubmed_qa_3branch"):
        raw = yaml.safe_load(open(os.path.join(root, f"configs/{name}.yaml")))
        cfg = EvaluationConfig.from_dict(build_evaluation_config_kwargs(raw), validate=False)
        pv, run = build_graph_for_config(cfg), build_run_shape(cfg)
        assert [n.id for n in pv.nodes] == [n.id for n in run.nodes], name
        for a, b in zip(pv.nodes, run.nodes):
            assert a.bindings == b.bindings, f"{name}:{a.id}"
            assert a.input_aliases == b.input_aliases, f"{name}:{a.id}"
            assert a.depends_on == b.depends_on, f"{name}:{a.id}"


def test_canvas_round_trip_rebuilds_identical_graph_for_every_config():
    """E5 gate: config → canvas → exportSpec-shaped spec → graph_spec_to_config_dict →
    rebuilt graph has the SAME bindings as the original (plumbing edges re-derived for
    appended structural nodes; canvas edges are authoritative for the meaningful ones)."""
    import glob
    import os

    import yaml

    from evaluator.config.evaluation import EvaluationConfig
    from evaluator.config.graph_config import build_evaluation_config_kwargs
    from evaluator.pipeline.graph.modes import build_graph_for_config
    from evaluator.webapi.form_builder import config_to_canvas_spec
    from evaluator.webapi.form_config import graph_spec_to_config_dict

    def _port_key(node, art):
        for port in node.get("input_ports") or []:
            if art in port["names"]:
                return port["label"]
        return art

    root = os.path.dirname(os.path.dirname(__file__))
    for path in sorted(glob.glob(os.path.join(root, "configs/*.yaml"))):
        raw = yaml.safe_load(open(path))
        if "branches" in (raw.get("graph") or {}):
            continue  # branch panel round-trips separately (variants ride the spec)
        if "datasets" in raw:
            continue  # multi-source: map keys need the datasets: block (pre-existing gap)
        name = os.path.basename(path)
        cfg = EvaluationConfig.from_dict(build_evaluation_config_kwargs(raw), validate=False)
        original = build_graph_for_config(cfg)
        canvas = config_to_canvas_spec(cfg)
        # exportSpec-shaped: ALL nodes (full-DAG view), port-level edges from bindings
        edges, seen = [], set()
        for n in canvas["nodes"]:
            for art, prod in n["bindings"]:
                tup = (prod, art, n["id"], _port_key(n, art))
                if tup not in seen:
                    seen.add(tup)
                    edges.append({"from": prod, "output": art,
                                  "to": n["id"], "input": tup[3]})
        spec = {"nodes": [{"id": n["id"], "type": n["type"], "params": dict(n["params"] or {})}
                          for n in canvas["nodes"]],
                "edges": edges}
        cfg2 = EvaluationConfig.from_dict(
            graph_spec_to_config_dict(spec, experiment_name="rt"), validate=False)
        rebuilt = build_graph_for_config(cfg2)
        a_by_id = {n.id: n for n in original.nodes}
        b_by_id = {n.id: n for n in rebuilt.nodes}
        # the plumbing derive may APPEND structural nodes an explicit config omitted
        # (documented Approach-B behavior); every original node must survive unchanged.
        assert set(a_by_id) <= set(b_by_id), name
        for nid, a in a_by_id.items():
            assert a.bindings == b_by_id[nid].bindings, f"{name}:{nid}"


def test_nodes_without_edges_rejected_at_load():
    """E6 hard cut: a config graph is nodes AND port-level edges; the error names the
    generator."""
    import pytest

    from evaluator.config.graph_config import GraphConfigError, build_evaluation_config_kwargs

    with pytest.raises(GraphConfigError, match="emit-edges"):
        build_evaluation_config_kwargs({"graph": {"nodes": ["dataset_source", "asr"]}})
    with pytest.raises(GraphConfigError, match="emit-edges"):
        # ordering-only edges don't satisfy explicitness either
        build_evaluation_config_kwargs({"graph": {"nodes": ["dataset_source", "asr"],
                                  "edges": [{"from": "dataset_source", "to": "asr"}]}})
