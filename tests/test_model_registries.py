"""Model family registries (asr / text_embedding / audio_embedding / reranker / tts)."""

import pytest

from evaluator.models.registry import FAMILY_REGISTRIES


def test_expected_families_exist():
    assert {"asr", "text_embedding", "audio_embedding", "reranker", "tts"} <= set(
        FAMILY_REGISTRIES
    )


@pytest.mark.parametrize(
    "family, expected",
    [
        ("asr", "whisper"),
        ("text_embedding", "jina_v4"),
        ("text_embedding", "labse"),
        ("audio_embedding", "attention_pool"),
        ("audio_embedding", "attention_pool_m4t"),
        ("audio_embedding", "sonar_speech"),
        ("audio_embedding", "eam_alignment"),
        ("text_embedding", "eam_alignment"),
    ],
)
def test_known_models_registered(family, expected):
    reg = FAMILY_REGISTRIES[family]
    assert expected in reg.list_types()
    assert reg.is_registered(expected)


def test_unknown_model_is_not_registered():
    assert not FAMILY_REGISTRIES["asr"].is_registered("not_a_real_model")


def test_resolve_model_name_by_size_and_default():
    audio = FAMILY_REGISTRIES["audio_embedding"]
    assert audio.resolve_model_name("attention_pool", size="large-v3") == (
        "openai/whisper-large-v3"
    )
    assert audio.resolve_model_name("attention_pool_m4t", size="v2-large") == (
        "facebook/seamless-m4t-v2-large"
    )
    # no size → the registered default name
    assert audio.get_default_name("attention_pool") == "openai/whisper-large"


def test_params_schema_exposes_pooling_choices():
    schema = FAMILY_REGISTRIES["audio_embedding"].get_params_schema("attention_pool")
    assert "pooling" in schema
    assert "attention" in schema["pooling"]["choices"]
    assert "mean_abtt" in schema["pooling"]["choices"]


def test_sonar_speech_declares_shared_embedding_space():
    # sonar speech + sonar text share a space → cross-modal retrieval is valid.
    meta = FAMILY_REGISTRIES["audio_embedding"].get_metadata("sonar_speech")
    assert meta.get("embedding_space") == "sonar"


def test_curated_llm_list_shape():
    # The picker's curated local list: unique ids, exactly the 5 fields the UI shows.
    from evaluator.models.llm import CURATED_LOCAL_MODELS

    assert CURATED_LOCAL_MODELS
    ids = [m["id"] for m in CURATED_LOCAL_MODELS]
    assert len(ids) == len(set(ids))
    for m in CURATED_LOCAL_MODELS:
        assert set(m) == {"id", "label", "domain", "size", "min_ram_gb"}


def test_memory_estimate_warns_on_unknown_model():
    # F26: a registered model returns its estimate silently; an unknown one warns + defaults.
    # Capture directly off the module logger — the project's loggers set propagate=False
    # once setup_logging runs, so pytest's caplog (root handler) can't see them.
    import io
    import logging

    from evaluator.config.base import estimate_model_memory_gb

    lg = logging.getLogger("evaluator.config.base")
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setLevel(logging.WARNING)
    old_level = lg.level
    lg.addHandler(handler)
    lg.setLevel(logging.WARNING)
    try:
        assert estimate_model_memory_gb("text_embedding", "labse") == 1.0  # known → silent
        assert "labse" not in buf.getvalue()
        assert estimate_model_memory_gb("text_embedding", "no_such_model") == 1.5  # default
        assert "no_such_model" in buf.getvalue()
    finally:
        lg.removeHandler(handler)
        lg.setLevel(old_level)
