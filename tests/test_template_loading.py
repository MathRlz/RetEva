"""Config loading after pipeline_mode removal: every config form resolves to a graph template
(`graph_override['template']`) or an explicit graph, and a bad template name errors clearly.
"""

import pytest

from evaluator.config.evaluation import EvaluationConfig
from evaluator.config.graph_config import GraphConfigError, build_evaluation_config_kwargs
from evaluator.pipeline.graph.modes import build_graph_for_config
from tests.graph_test_helpers import explicit_graph, mode_graph

_DATASET = {
    "id": "pubmed_qa",
    "questions": "examples/data/pubmed_qa_small/questions.json",
    "corpus": "examples/data/pubmed_qa_small/corpus.json",
}


def test_legacy_flat_pipeline_mode_loads_as_template():
    # A legacy flat `model: {pipeline_mode: X}` config (no graph block) must still load — the loader
    # moves it onto graph_override['template'] before ModelConfig construction.
    cfg = EvaluationConfig.from_dict(
        {"model": {"pipeline_mode": "audio_emb_retrieval",
                   "audio_emb_model_type": "attention_pool"}},
        validate=False,
    )
    assert cfg.graph_template == "audio_emb_retrieval"
    assert getattr(cfg.model, "pipeline_mode", None) is None
    assert cfg.model.audio_emb_model_type == "attention_pool"


def test_graph_mode_in_a_config_is_rejected():
    # Hard cut-over: a config is an explicit graph — graph.mode is no longer accepted.
    with pytest.raises(GraphConfigError, match="graph.mode is no longer supported"):
        build_evaluation_config_kwargs({"dataset": _DATASET, "graph": {"mode": "asr_text_retrieval"}})


def test_features_block_in_a_config_is_rejected():
    # Hard cut-over: a capability is enabled by its node, not a features: block.
    from evaluator.errors import ConfigurationError

    with pytest.raises(ConfigurationError, match="features:` block is no longer supported"):
        EvaluationConfig.from_dict({"features": {"judge": {"enabled": True}}}, validate=False)


def test_graph_nodes_form_is_graph_only():
    cfg = EvaluationConfig.from_dict(
        build_evaluation_config_kwargs({"dataset": _DATASET, "graph": explicit_graph([
            {"id": "dataset_source", "type": "dataset_source"},
            {"id": "asr", "type": "asr", "params": {"model": "whisper"}},
            {"id": "text_embedding", "type": "text_embedding", "params": {"model": "labse"}},
            {"id": "corpus_embedding", "type": "corpus_embedding"},
            {"id": "vector_db", "type": "vector_db", "params": {"store": "inmemory"}},
            {"id": "retrieval", "type": "retrieval", "params": {"k": 5}},
        ])}), validate=False,
    )
    assert cfg.graph_template == "asr_text_retrieval"   # label derived from the explicit graph
    assert not (cfg.graph_override or {}).get("template")  # no mode template, just nodes
    assert cfg.graph_override.get("nodes")


def test_explicit_graph_round_trips_through_from_dict():
    # graph_override survives a to_dict(include_config=True) → from_dict round-trip (audit A3).
    src = EvaluationConfig.from_dict(
        build_evaluation_config_kwargs({"dataset": _DATASET,
                        "graph": mode_graph("asr_text_retrieval")}), validate=False
    )
    nested = src.to_dict(include_config=True)   # the round-trippable nested config dict
    rt = EvaluationConfig.from_dict(nested, validate=False)
    assert rt.graph_template == "asr_text_retrieval"


def test_unknown_template_errors_clearly_at_build():
    cfg = EvaluationConfig.from_dict(
        {"graph_override": {"template": "asr_text_retrival"}}, validate=False)
    with pytest.raises(ValueError, match="Unknown graph template"):
        build_graph_for_config(cfg)


def test_unknown_template_caught_at_validate_time():
    # Audit D: a typo'd template is rejected at validate() with the available list, not only
    # later at graph-build.
    from evaluator.errors import ConfigurationError
    with pytest.raises(ConfigurationError, match="unknown graph template"):
        EvaluationConfig.from_dict({"graph_override": {"template": "bogus"}}, validate=True)
