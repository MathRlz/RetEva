"""The resolved config is the *executed DAG*, node-centric, no mode.

The resolved config serializes the graph that actually ran — so a node (and its model) the run
never executed can never appear (e.g. an ``audio_emb_retrieval`` run must never show
``asr_model: wav2vec2:default`` when ASR never ran) — and round-trips it.
"""

import copy

from evaluator.config.evaluation import EvaluationConfig
from evaluator.config.serialization import resolved_node_config
from evaluator.config.graph_config import build_evaluation_config_kwargs
from evaluator.pipeline.graph.build import build_graph_for_config
from tests.fixtures_paths import E2E_MINI, MINI_C, MINI_Q

_DATASET = {
    "id": "pubmed_qa",
    "questions": MINI_Q,
    "corpus": MINI_C,
}


def _cfg(graph):
    # deepcopy: build_evaluation_config_kwargs mutates its input (folds + empties node params), and the graph
    # fixtures are shared module-level dicts.
    from evaluator.pipeline.graph.wiring import emit_edges

    payload = copy.deepcopy({"dataset": _DATASET, "graph": graph})
    g = payload["graph"]
    g.setdefault("edges", emit_edges(g["nodes"]))
    return EvaluationConfig.from_dict(build_evaluation_config_kwargs(payload), validate=False)


_AUDIO_EMB = {
    "nodes": [
        {"id": "dataset_source", "type": "dataset_source"},
        {"id": "audio_embedding", "type": "audio_embedding",
         "params": {"model": "attention_pool", "dim": 1024}},
        {"id": "corpus_embedding", "type": "corpus_embedding"},
        {"id": "vector_db", "type": "vector_db", "params": {"store": "inmemory"}},
        {"id": "retrieval", "type": "retrieval", "params": {"k": 5}},
    ],
}


def test_audio_emb_resolved_config_has_no_asr_and_no_mode():
    rc = resolved_node_config(_cfg(_AUDIO_EMB))
    kinds = [n["type"] for n in rc["graph"]["nodes"]]
    assert "asr" not in kinds                    # the run had no ASR node
    assert "audio_embedding" in kinds
    # No default asr model leaks anywhere, and there is no pipeline_mode (the graph is the spec).
    import json
    blob = json.dumps(rc)
    assert "asr_model" not in blob
    assert "pipeline_mode" not in blob
    assert "mode" not in rc["graph"]


def test_resolved_model_node_carries_only_resolved_params():
    rc = resolved_node_config(_cfg(_AUDIO_EMB))
    audio = next(n for n in rc["graph"]["nodes"] if n["id"] == "audio_embedding")
    assert audio["params"]["model"] == "attention_pool"
    assert audio["params"]["dim"] == 1024
    # unset fields are pruned (no None noise)
    assert None not in audio["params"].values()
    assert "size" not in audio["params"]


def test_resolved_config_round_trips():
    cfg = _cfg(_AUDIO_EMB)
    rc = resolved_node_config(cfg)
    # Load the resolved config back → rebuild the graph → identical node set + resolved model.
    cfg2 = EvaluationConfig.from_dict(build_evaluation_config_kwargs(rc), validate=False)
    graph2 = build_graph_for_config(cfg2)
    ids1 = [n.id for n in build_graph_for_config(cfg).nodes]
    ids2 = [n.id for n in graph2.nodes]
    assert ids1 == ids2
    # The audio_embedding node's own model params round-trip as a per-node override (they
    # never get promoted into the flat default — resolve it the way the executor does).
    from evaluator.config.graph_config import resolved_model_config

    node2 = next(n for n in graph2.nodes if n.id == "audio_embedding")
    resolved = resolved_model_config(cfg2, node2)
    assert resolved.audio_emb_model_type == "attention_pool"
    assert resolved.audio_emb_dim == 1024
    assert cfg2.graph_label == "audio_emb_retrieval"   # label derived from the graph nodes
    assert cfg2.vector_db.k == 5


def test_asr_text_resolved_config_lists_asr_text_retrieval():
    rc = resolved_node_config(_cfg({
        "nodes": [
            {"id": "dataset_source", "type": "dataset_source"},
            {"id": "asr", "type": "asr",
             "params": {"model": "whisper", "name": "openai/whisper-base"}},
            {"id": "text_embedding", "type": "text_embedding", "params": {"model": "labse"}},
            {"id": "corpus_embedding", "type": "corpus_embedding"},
            {"id": "vector_db", "type": "vector_db", "params": {"store": "inmemory"}},
            {"id": "retrieval", "type": "retrieval", "params": {"k": 5}},
        ],
    }))
    kinds = {n["type"] for n in rc["graph"]["nodes"]}
    assert {"asr", "text_embedding", "retrieval"} <= kinds
    asr = next(n for n in rc["graph"]["nodes"] if n["id"] == "asr")
    assert asr["params"]["model"] == "whisper"
    assert asr["params"]["name"] == "openai/whisper-base"


def test_resolved_config_round_trips_structural_params():
    """The resolved sidecar must rebuild the same executed DAG for a plain
    explicit-graph config (no graph.branches — comparing variants is just distinctly-named
    nodes in the one graph.nodes list, nothing to serialize pre-expansion). Being
    self-contained, the sidecar inlines onto each node what the source left to a `nodes:` /
    `model:` default; it must never drop or change a param, nor move an edge."""
    cfg = EvaluationConfig.from_yaml(E2E_MINI, validate=False)
    g1 = build_graph_for_config(cfg)
    rc = resolved_node_config(cfg)
    cfg2 = EvaluationConfig.from_dict(build_evaluation_config_kwargs(rc), validate=False)
    g2 = build_graph_for_config(cfg2)

    rebuilt = {n.id: n for n in g2.nodes}
    assert set(rebuilt) == {n.id for n in g1.nodes}, "E2E_MINI: node set changed"
    for n in g1.nodes:
        other = rebuilt[n.id]
        assert sorted(other.bindings) == sorted(n.bindings), f"{n.id}: edges changed"
        assert (n.params or {}).items() <= (other.params or {}).items(), f"{n.id}: params lost"
    # the config's `nodes:` block resolves ONTO the nodes it configures
    assert rebuilt["vector_db"].params["store"] == "inmemory"
    assert rebuilt["asr"].params["model"] == "whisper"
    assert rebuilt["asr"].params["name"] == "openai/whisper-base"
    assert rebuilt["text_embedding"].params["model"] == "labse"
