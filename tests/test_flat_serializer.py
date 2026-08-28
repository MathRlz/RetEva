"""Flat `to_dict(include_config=False)` telemetry emits a model summary only for the families the
executed graph uses — the original bug was a default `asr_model: wav2vec2:default` for a run that
never had an ASR node (audit / graph-first Phase 3)."""

from evaluator.config.evaluation import EvaluationConfig
from evaluator.config.graph_config import build_evaluation_config_kwargs
from tests.graph_test_helpers import mode_graph

_DATASET = {
    "id": "pubmed_qa",
    "questions": "examples/data/pubmed_qa_small/questions.json",
    "corpus": "examples/data/pubmed_qa_small/corpus.json",
}


def _model_keys(template):
    cfg = EvaluationConfig.from_dict(
        build_evaluation_config_kwargs({"dataset": _DATASET, "graph": mode_graph(template)}),
        validate=False,
    )
    return {k for k in cfg.to_dict() if k.endswith("_model")}


def test_asr_text_emits_asr_and_text_only():
    assert _model_keys("asr_text_retrieval") == {"asr_model", "text_emb_model"}


def test_audio_emb_emits_audio_only_no_asr():
    keys = _model_keys("audio_emb_retrieval")
    assert "audio_emb_model" in keys
    assert "asr_model" not in keys           # the run had no ASR node — the original bug


def test_asr_only_emits_asr_only():
    keys = _model_keys("asr_only")
    assert keys == {"asr_model"}


def test_flat_dict_pipeline_mode_is_the_derived_label():
    # Audit 2026-07 #3: the flat config echo (stored in leaderboard rows) must carry the
    # derived graph label, not None (graph_override['template'] is never set post-cut-over).
    cfg = EvaluationConfig.from_dict(
        build_evaluation_config_kwargs({"dataset": _DATASET,
                        "graph": mode_graph("asr_text_retrieval")}),
        validate=False,
    )
    assert cfg.to_dict()["pipeline_mode"] == "asr_text_retrieval"
