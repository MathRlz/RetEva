"""Config presets: discoverable + each loads into a valid EvaluationConfig."""

import pytest

from evaluator.config.model_presets import list_presets


def test_presets_are_discoverable():
    presets = list_presets()
    assert presets, "no presets found in configs/"
    assert "e2e_pubmed_qa_small" in presets


@pytest.mark.parametrize(
    "name", ["e2e_pubmed_qa_small", "evaluation_config_admed_selfretr_asr_text"]
)
def test_named_preset_loads(name):
    from evaluator.config.evaluation import EvaluationConfig

    if name not in list_presets():
        pytest.skip(f"{name} not present")
    cfg = EvaluationConfig.from_preset(name, validate=False)
    assert cfg.graph_template is not None


@pytest.mark.parametrize(
    "key,expected",
    [
        ("model_asr_device", ("model", "asr_device")),
        ("model_asr_model_name", ("model", "asr_model_name")),
        ("data_batch_size", ("data", "batch_size")),
        ("cache_enabled", ("cache", "enabled")),
        ("vector_db_k", ("vector_db", "k")),
        ("vector_db_retrieval_mode", ("vector_db", "retrieval_mode")),
        ("device_pool_enabled", ("device_pool", "enabled")),
        ("service_runtime_offload_policy", ("service_runtime", "offload_policy")),
        ("audio_synthesis_seed", ("audio_synthesis", "seed")),
        # not a multi-word section → plain top-level keys
        ("vector_foo", (None, "vector_foo")),
        ("device_foo", (None, "device_foo")),
        ("audio_foo", (None, "audio_foo")),
        ("experiment_name", (None, "experiment_name")),
        ("experiment", (None, "experiment")),
    ],
)
def test_underscore_override_resolution(key, expected):
    # F13: the generic resolver replaces the hand-sliced prefix decoder.
    from evaluator.config.loading import _resolve_underscore_override

    assert _resolve_underscore_override(key) == expected


def test_from_preset_applies_underscore_and_dotted_overrides():
    from evaluator.config.evaluation import EvaluationConfig

    if "e2e_pubmed_qa_small" not in list_presets():
        pytest.skip("preset not present")
    cfg = EvaluationConfig.from_preset(
        "e2e_pubmed_qa_small", validate=False,
        model_asr_device="cpu",              # underscore → model.asr_device
        vector_db_k=11,                      # underscore multi-word → vector_db.k
    )
    assert cfg.model.asr_device == "cpu"
    assert cfg.vector_db.k == 11
