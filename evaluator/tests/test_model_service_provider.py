"""Tests for model service provider lifecycle behavior."""

import logging

from evaluator.config import AudioSynthesisConfig, EvaluationConfig, LLMServerConfig
from evaluator.pipeline import create_pipeline_from_config
from evaluator.storage.cache import CacheManager
from evaluator.services.model_provider import ModelServiceProvider


class _DummyModel:
    def __init__(self):
        self.moves = []

    def to(self, device):
        self.moves.append(str(device))
        return self


def test_provider_reuses_same_asr_service(monkeypatch):
    created = {"count": 0}

    def _fake_create_asr_model(**_kwargs):
        created["count"] += 1
        return _DummyModel()

    monkeypatch.setattr(
        "evaluator.services.model_provider.factory_create_asr_model",
        _fake_create_asr_model,
    )

    provider = ModelServiceProvider()
    m1 = provider.get_asr_model("whisper", "openai/whisper-small", None, "cpu")
    m2 = provider.get_asr_model("whisper", "openai/whisper-small", None, "cpu")

    assert m1 is m2
    assert created["count"] == 1


def test_provider_shutdown_offloads_models(monkeypatch):
    model = _DummyModel()

    monkeypatch.setattr(
        "evaluator.services.model_provider.factory_create_text_embedding_model",
        lambda **_kwargs: model,
    )
    monkeypatch.setattr(
        "evaluator.services.model_services.torch.cuda.is_available",
        lambda: False,
    )

    provider = ModelServiceProvider()
    provider.get_text_embedding_model("labse", "sentence-transformers/LaBSE", "cpu")
    provider.shutdown()

    assert "cpu" in model.moves


def test_provider_move_model_between_devices(monkeypatch):
    model = _DummyModel()

    monkeypatch.setattr(
        "evaluator.services.model_provider.factory_create_asr_model",
        lambda **_kwargs: model,
    )

    provider = ModelServiceProvider()
    provider.get_asr_model("whisper", "openai/whisper-small", None, "cuda:0")
    moved = provider.move_asr_model(
        "whisper",
        "openai/whisper-small",
        None,
        "cuda:0",
        "cuda:1",
    )

    assert moved is model
    assert "cuda:1" in model.moves


def test_provider_release_single_model(monkeypatch):
    created = {"count": 0}

    def _fake_create_audio_embedding_model(**_kwargs):
        created["count"] += 1
        return _DummyModel()

    monkeypatch.setattr(
        "evaluator.services.model_provider.factory_create_audio_embedding_model",
        _fake_create_audio_embedding_model,
    )
    monkeypatch.setattr(
        "evaluator.services.model_services.torch.cuda.is_available",
        lambda: False,
    )

    provider = ModelServiceProvider()
    provider.get_audio_embedding_model(
        "attention_pool",
        "openai/whisper-base",
        None,
        2048,
        0.1,
        "cpu",
    )
    provider.release_audio_embedding_model(
        "attention_pool",
        "openai/whisper-base",
        None,
        2048,
        0.1,
        "cpu",
    )
    provider.get_audio_embedding_model(
        "attention_pool",
        "openai/whisper-base",
        None,
        2048,
        0.1,
        "cpu",
    )

    assert created["count"] == 2


def test_provider_lists_available_models():
    provider = ModelServiceProvider()
    models = provider.list_available_models()

    assert "asr" in models
    assert "text_embedding" in models
    assert "audio_embedding" in models
    assert "reranker" in models
    assert "tts" in models
    assert any(entry["type"] == "whisper" for entry in models["asr"])
    assert any(entry["type"] == "labse" for entry in models["text_embedding"])
    assert any(entry["type"] == "attention_pool" for entry in models["audio_embedding"])
    assert any(entry["type"] == "cross_encoder" for entry in models["reranker"])
    assert any(entry["type"] == "xtts_v2" for entry in models["tts"])


def test_provider_emits_lifecycle_logs(monkeypatch, caplog):
    model = _DummyModel()
    monkeypatch.setattr(
        "evaluator.services.model_provider.factory_create_asr_model",
        lambda **_kwargs: model,
    )
    monkeypatch.setattr(
        "evaluator.services.model_services.torch.cuda.is_available",
        lambda: False,
    )

    provider = ModelServiceProvider()
    with caplog.at_level(logging.INFO):
        provider.get_asr_model("whisper", "openai/whisper-small", None, "cpu")
        provider.move_asr_model("whisper", "openai/whisper-small", None, "cpu", "cpu")
        provider.release_asr_model("whisper", "openai/whisper-small", None, "cpu")

    text = " ".join(caplog.messages)
    assert "service.start" in text
    assert "provider.move" in text
    assert "provider.release" in text


def test_pipeline_bundle_uses_movable_service_models(monkeypatch):
    created = {"asr": 0, "text": 0}

    class _DummyASR(_DummyModel):
        def name(self):
            return "dummy-asr"

    class _DummyText(_DummyModel):
        def name(self):
            return "dummy-text"

    def _new_asr(**_kwargs):
        created["asr"] += 1
        return _DummyASR()

    def _new_text(**_kwargs):
        created["text"] += 1
        return _DummyText()

    monkeypatch.setattr(
        "evaluator.services.model_provider.factory_create_asr_model",
        _new_asr,
    )
    monkeypatch.setattr(
        "evaluator.services.model_provider.factory_create_text_embedding_model",
        _new_text,
    )
    monkeypatch.setattr(
        "evaluator.services.model_services.torch.cuda.is_available",
        lambda: False,
    )

    config = EvaluationConfig()
    config.model.pipeline_mode = "asr_text_retrieval"
    config.model.asr_model_type = "whisper"
    config.model.asr_model_name = None
    config.model.text_emb_model_type = "labse"
    config.model.text_emb_model_name = None
    config.model.asr_adapter_path = None
    config.model.asr_device = "cuda:0"
    config.model.text_emb_device = "cuda:0"
    config.vector_db.reranker_enabled = False

    provider = ModelServiceProvider()
    cache = CacheManager(enabled=False)
    bundle = create_pipeline_from_config(config, cache, service_provider=provider)
    asr_model = bundle.asr_pipeline.model

    provider.move_asr_model("whisper", None, None, "cuda:0", "cpu")
    assert asr_model is bundle.asr_pipeline.model
    assert "cpu" in asr_model.moves

    provider.release_asr_model("whisper", None, None, "cpu")
    provider.get_asr_model("whisper", None, None, "cpu")
    assert created["asr"] == 2


def test_provider_shutdown_without_offload(monkeypatch):
    model = _DummyModel()
    monkeypatch.setattr(
        "evaluator.services.model_provider.factory_create_asr_model",
        lambda **_kwargs: model,
    )

    provider = ModelServiceProvider()
    provider.get_asr_model("whisper", "openai/whisper-small", None, "cuda:0")
    provider.shutdown(offload=False)

    assert model.moves == []


def test_provider_manages_tts_service(monkeypatch):
    created = {"count": 0}

    class _DummyTTS(_DummyModel):
        def synthesize(self, text):
            return text

    def _create_tts(_cfg):
        created["count"] += 1
        return _DummyTTS()

    monkeypatch.setattr(
        "evaluator.services.model_provider.ModelServiceProvider._create_tts_backend",
        staticmethod(_create_tts),
    )
    monkeypatch.setattr(
        "evaluator.services.model_services.torch.cuda.is_available",
        lambda: False,
    )

    cfg = AudioSynthesisConfig(enabled=True, provider="mms", language="en", voice="v1")
    provider = ModelServiceProvider()
    t1 = provider.get_tts_model(cfg)
    t2 = provider.get_tts_model(cfg)
    assert t1 is t2
    assert created["count"] == 1

    provider.release_tts_model(cfg)
    t3 = provider.get_tts_model(cfg)
    assert t3 is not t1
    assert created["count"] == 2


def test_pipeline_service_mode_asr_only(monkeypatch):
    class _DummyASR(_DummyModel):
        def name(self):
            return "dummy-asr"

    monkeypatch.setattr(
        "evaluator.services.model_provider.factory_create_asr_model",
        lambda **_kwargs: _DummyASR(),
    )
    monkeypatch.setattr(
        "evaluator.services.model_services.torch.cuda.is_available",
        lambda: False,
    )

    config = EvaluationConfig()
    config.model.pipeline_mode = "asr_only"
    config.model.asr_model_type = "whisper"
    config.model.asr_device = "cpu"

    provider = ModelServiceProvider()
    cache = CacheManager(enabled=False)
    bundle = create_pipeline_from_config(config, cache, service_provider=provider)

    assert bundle.asr_pipeline is not None
    assert bundle.text_embedding_pipeline is None
    assert bundle.audio_embedding_pipeline is None
    assert bundle.retrieval_pipeline is None


def test_pipeline_service_mode_audio_embedding(monkeypatch):
    class _DummyAudio(_DummyModel):
        def name(self):
            return "dummy-audio"

    monkeypatch.setattr(
        "evaluator.services.model_provider.factory_create_audio_embedding_model",
        lambda **_kwargs: _DummyAudio(),
    )
    monkeypatch.setattr(
        "evaluator.services.model_services.torch.cuda.is_available",
        lambda: False,
    )

    config = EvaluationConfig()
    config.model.pipeline_mode = "audio_emb_retrieval"
    config.model.audio_emb_model_type = "attention_pool"
    config.model.audio_emb_model_name = None
    config.model.audio_emb_model_path = None
    config.model.audio_emb_device = "cpu"
    config.vector_db.reranker_enabled = False

    provider = ModelServiceProvider()
    cache = CacheManager(enabled=False)
    bundle = create_pipeline_from_config(config, cache, service_provider=provider)

    assert bundle.audio_embedding_pipeline is not None
    assert bundle.retrieval_pipeline is not None


def test_provider_manages_local_llm_server(monkeypatch):
    created = {"count": 0}

    class _Health:
        def __init__(self, is_healthy):
            self.is_healthy = is_healthy

    class _DummyServer:
        def __init__(self):
            self.running = False
            self.stop_calls = 0

        def health_check(self):
            return _Health(self.running)

        def start(self):
            self.running = True
            return True

        def stop(self):
            self.stop_calls += 1
            self.running = False

        def get_api_url(self):
            return "http://localhost:11434/v1/chat/completions"

    def _create_server(**_kwargs):
        created["count"] += 1
        return _DummyServer()

    monkeypatch.setattr(
        "evaluator.services.model_provider.factory_create_llm_server",
        _create_server,
    )

    provider = ModelServiceProvider()
    cfg = LLMServerConfig(enabled=True, backend="ollama", model="mistral:7b-instruct")
    s1 = provider.get_llm_server(cfg)
    s2 = provider.get_llm_server(cfg)

    assert s1 is s2
    assert created["count"] == 1

    provider.release_llm_server(cfg)
    s3 = provider.get_llm_server(cfg)
    assert s3 is not s1
    assert created["count"] == 2


def test_provider_does_not_stop_external_llm_server_on_shutdown(monkeypatch):
    class _Health:
        def __init__(self, is_healthy):
            self.is_healthy = is_healthy

    class _DummyServer:
        def __init__(self):
            self.stop_calls = 0

        def health_check(self):
            return _Health(True)  # already running externally

        def start(self):
            return True

        def stop(self):
            self.stop_calls += 1

        def get_api_url(self):
            return "http://localhost:11434/v1/chat/completions"

    holder = {}

    def _create_server(**_kwargs):
        holder["server"] = _DummyServer()
        return holder["server"]

    monkeypatch.setattr(
        "evaluator.services.model_provider.factory_create_llm_server",
        _create_server,
    )

    provider = ModelServiceProvider()
    cfg = LLMServerConfig(enabled=True, backend="ollama", model="mistral:7b-instruct")
    provider.get_llm_server(cfg)
    provider.shutdown()

    assert holder["server"].stop_calls == 0
