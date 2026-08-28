"""Pre-flight model-existence validation (`validate_models`): the config is checked before any
model loads — an unregistered model type or a missing adapter path fails fast with a clear
ConfigurationError; valid configs pass; only the models the graph builds are checked."""
import pytest

from evaluator.config import EvaluationConfig
from evaluator.errors import ConfigurationError
from evaluator.evaluation.validation import validate_models
from evaluator.pipeline import build_graph_for_config


def _graph(cfg):
    return build_graph_for_config(cfg)


def _minimal():
    # whisper + labse via the legacy template reference (the back-compat path under test)
    cfg = EvaluationConfig.from_dict({}, validate=False)
    cfg.graph_override = {"template": "asr_text_retrieval"}
    cfg.model.asr_model_type = "whisper"
    cfg.model.text_emb_model_type = "labse"
    return cfg


def _audio_only():
    cfg = EvaluationConfig.from_dict({}, validate=False)
    cfg.graph_override = {"template": "audio_emb_retrieval"}
    cfg.model.audio_emb_model_type = "attention_pool"
    cfg.model.text_emb_model_type = "labse"
    cfg.model.audio_emb_dim = 768
    return cfg


def test_valid_models_pass():
    cfg = _minimal()  # whisper + labse, both registered
    validate_models(cfg, _graph(cfg))  # must not raise


def test_unregistered_type_raises_with_available_list():
    cfg = _minimal()
    cfg.model.asr_model_type = "whisperr"  # typo
    with pytest.raises(ConfigurationError) as exc:
        validate_models(cfg, _graph(cfg))
    msg = str(exc.value)
    assert "asr_model_type" in msg and "whisperr" in msg
    assert "available:" in msg and "whisper" in msg  # lists the real options


def test_missing_adapter_path_raises():
    cfg = _minimal()
    cfg.model.text_emb_adapter_path = "/nope/does-not-exist.pth"
    with pytest.raises(ConfigurationError, match="text_emb_adapter_path not found"):
        validate_models(cfg, _graph(cfg))


def test_unused_family_default_not_checked():
    # audio_emb run: no asr node runs, so a bogus asr type in the config is NOT validated —
    # only the models the graph builds (audio_emb + text corpus) are checked.
    cfg = _audio_only()
    cfg.model.asr_model_type = "bogus_unused"
    validate_models(cfg, _graph(cfg))  # must not raise


def test_audio_and_text_families_validated():
    cfg = _audio_only()
    cfg.model.audio_emb_model_type = "not_a_real_audio_model"
    with pytest.raises(ConfigurationError, match="audio_emb_model_type"):
        validate_models(cfg, _graph(cfg))
