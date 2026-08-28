"""Phase 4.6 — no-model integration: cover the builder→run path *past* the fake-runner boundary.

The from-graph endpoint tests stub the runner (they stop at submit). These exercise the real
chain spec → config → built DAG → resolved node config (pure topology + config, no models
loaded), so the translation that feeds the executor is verified end to end.
"""

from pathlib import Path

import pytest

from evaluator.config.graph_config import resolved_node_config
from evaluator.pipeline import build_graph_for_config
from evaluator.pipeline.graph.operators import node_kind
from evaluator.webapi.form_config import graph_spec_to_config_dict, prepare_run_config

_ROOT = Path(__file__).resolve().parents[1]
_Q = str(_ROOT / "examples/data/pubmed_qa_small/questions.json")
_C = str(_ROOT / "examples/data/pubmed_qa_small/corpus.json")


def _spec():
    return {"mode": "asr_text_retrieval", "nodes": [
        {"id": "dataset_source", "type": "source",
         "params": {"dataset": "pubmed_qa", "questions": _Q, "corpus": _C}},
        {"id": "asr", "type": "convert",
         "params": {"model": "whisper", "name": "openai/whisper-base"}},
        {"id": "text_embedding", "type": "embed", "params": {"model": "labse"}},
        {"id": "corpus_embedding", "type": "embed", "params": {"axis": "corpus"}},
        {"id": "vector_db", "type": "index", "params": {}},
        {"id": "retrieval", "type": "search", "params": {}},
        {"id": "metrics", "type": "measure", "params": {}},
        {"id": "finalize", "type": "sink", "params": {}},
    ], "edges": [
        {"from": "dataset_source", "to": "asr"}, {"from": "asr", "to": "text_embedding"},
        {"from": "dataset_source", "to": "corpus_embedding"},
        {"from": "corpus_embedding", "to": "vector_db"},
        {"from": "text_embedding", "to": "retrieval"},
        {"from": "vector_db", "to": "retrieval"},
        {"from": "retrieval", "to": "metrics"}, {"from": "metrics", "to": "finalize"},
    ]}


def test_spec_builds_a_runnable_config():
    """spec → config carries the models + dataset the graph declared (a node's own model params
    stay ON the node — a per-node override — resolved the way the executor does)."""
    from evaluator.config.graph_config import resolved_model_config

    cfg = prepare_run_config(graph_spec_to_config_dict(_spec()), auto_devices=True)
    graph = build_graph_for_config(cfg)
    asr_node = next(n for n in graph.nodes if node_kind(n.stage, n.params) == "asr")
    text_node = next(n for n in graph.nodes if node_kind(n.stage, n.params) == "text_embedding")
    asr_model = resolved_model_config(cfg, asr_node)
    text_model = resolved_model_config(cfg, text_node)
    assert asr_model.asr_model_type == "whisper"
    assert asr_model.asr_model_name == "openai/whisper-base"
    assert text_model.text_emb_model_type == "labse"
    assert "dataset_source" in (cfg.data.datasets or {})


def test_config_builds_the_expected_dag():
    """The config builds a DAG whose node kinds match the graph the user drew."""
    cfg = prepare_run_config(graph_spec_to_config_dict(_spec()), auto_devices=True)
    graph = build_graph_for_config(cfg)
    kinds = {node_kind(n.stage, n.params) for n in graph.nodes}
    assert {"asr", "text_embedding", "corpus_embedding", "vector_db", "retrieval"} <= kinds


def test_resolved_node_config_round_trips_the_models():
    """resolved_node_config (what the executor serializes) carries the resolved models on their
    nodes — the asr node lists whisper, proving the spec→executor wiring is intact."""
    cfg = prepare_run_config(graph_spec_to_config_dict(_spec()), auto_devices=True)
    resolved = resolved_node_config(cfg)
    graph_nodes = (resolved.get("graph") or {}).get("nodes") or []
    asr = next((n for n in graph_nodes
                if node_kind(n.get("type"), n.get("params") or {}) == "asr"), None)
    assert asr is not None
    assert (asr.get("params") or {}).get("model") == "whisper"


def test_unknown_dataset_is_rejected_at_prep():
    """A dataset_source naming an unregistered dataset fails at translation (not at run)."""
    from evaluator.config.graph_config import GraphConfigError

    spec = _spec()
    spec["nodes"][0]["params"] = {"dataset": "no_such_dataset"}
    with pytest.raises(GraphConfigError):
        graph_spec_to_config_dict(spec)


def test_dataset_less_source_falls_back_at_translation():
    """At the pure-translation level a dataset_source with no `dataset` doesn't synthesize a
    source — it would fall back to the config default. (The *Run* path guards against this
    explicitly: see test_run_from_graph_requires_dataset_choice — but the translator stays lenient
    so a config with a top-level `dataset:` + a bare source node still loads.)"""
    spec = _spec()
    spec["nodes"][0]["params"] = {}  # no dataset chosen
    cfg = prepare_run_config(graph_spec_to_config_dict(spec), auto_devices=True)
    assert not (cfg.data.datasets or {})  # no per-node dataset map synthesized
    assert cfg.data.dataset_name  # falls back to the config default name
