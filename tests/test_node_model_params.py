"""Phase 1 (graph-first): a node's per-instance params fully specify its model.

The per-node model build must overlay the node's params onto config.model and honor the FULL set
(model_path / dim / pooling-via-params / embedding_space / quantization / size), not just
model/name/device — so a graph node can own its model end-to-end.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import evaluator.pipeline.factory as factory_mod
from evaluator.config.evaluation import EvaluationConfig
from evaluator.evaluation.executor.node_pipeline import _build_node_pipeline, _node_pipeline


def _capture_effective_model(monkeypatch):
    captured = {}

    class _FakeBuilders:
        def __init__(self, config, provider, pool):
            captured["model"] = config.model
            captured["provider"] = provider
            captured["pool"] = pool

        def asr(self):
            return MagicMock()

        text_emb = audio_emb = asr

    monkeypatch.setattr(factory_mod, "_ModelBuilders", _FakeBuilders)
    # The pipeline wrappers just store (model, cache); a SimpleNamespace model is fine.
    return captured


def _state(service_provider=None, device_pool=None):
    return SimpleNamespace(
        config=EvaluationConfig(), cache_manager=None,
        current_node=SimpleNamespace(id="n"), service_provider=service_provider,
        device_pool=device_pool,
    )


def test_audio_node_params_fully_specify_the_model(monkeypatch):
    captured = _capture_effective_model(monkeypatch)
    params = {
        "model": "attention_pool", "name": "openai/whisper-large", "size": "large",
        "model_path": "/ckpt.pt", "dim": 6,
        "params": {"pooling": "mean_abtt"}, "embedding_space": "jina_space",
        "quantization": "int8", "device": "cpu",
    }
    _build_node_pipeline(_state(), "audio_embedding", params)
    m = captured["model"]
    assert m.audio_emb_model_type == "attention_pool"
    assert m.audio_emb_model_name == "openai/whisper-large"
    assert m.audio_emb_model_path == "/ckpt.pt"
    assert m.audio_emb_dim == 6
    assert m.audio_emb_params == {"pooling": "mean_abtt"}   # pooling reaches the model
    assert m.audio_emb_embedding_space == "jina_space"
    assert m.audio_emb_quantization == "int8"
    assert m.audio_emb_device == "cpu"
    # no shared provider on this state (default) => factory branch, same as before
    assert captured["provider"] is None


def test_node_pipeline_routes_through_the_shared_service_provider(monkeypatch):
    # R4: a per-node override now builds via the run's service_provider (when present) instead
    # of always forcing the standalone factory branch — so two nodes resolving to the SAME
    # model share one loaded instance via the provider's cache, instead of each paying for
    # its own load.
    captured = _capture_effective_model(monkeypatch)
    provider = object()
    _build_node_pipeline(
        _state(service_provider=provider), "text_embedding", {"model": "labse"}
    )
    assert captured["provider"] is provider


def test_text_node_params_overlay_only_set_fields(monkeypatch):
    captured = _capture_effective_model(monkeypatch)
    base = EvaluationConfig().model
    _build_node_pipeline(_state(), "text_embedding", {"model": "labse", "device": "cuda:1"})
    m = captured["model"]
    assert m.text_emb_model_type == "labse" and m.text_emb_device == "cuda:1"
    # unset fields keep the global defaults (overlay, not wipe)
    assert m.text_emb_model_name == base.text_emb_model_name


def test_two_nodes_with_the_same_model_share_one_loaded_instance(monkeypatch):
    # R4 end-to-end: no _ModelBuilders mocking here — a real ModelServiceProvider, only the
    # underlying model factory faked (so no actual weights load). Two nodes resolving to the
    # SAME (type, name, device) reuse the one loaded model; a node with a different model gets
    # its own.
    from evaluator.services.model_provider import ModelServiceProvider
    import evaluator.services.model_provider as model_provider_mod

    built = []

    def _fake_factory(*, model_type, model_name, device, **kwargs):
        m = SimpleNamespace(model_type=model_type, model_name=model_name, device=device)
        built.append(m)
        return m

    monkeypatch.setattr(
        model_provider_mod, "factory_create_text_embedding_model", _fake_factory
    )
    provider = ModelServiceProvider()

    p1 = _build_node_pipeline(
        _state(service_provider=provider), "text_embedding", {"model": "labse"}
    )
    p2 = _build_node_pipeline(
        _state(service_provider=provider), "text_embedding", {"model": "labse"}
    )
    p3 = _build_node_pipeline(
        _state(service_provider=provider), "text_embedding", {"model": "jina_v4"}
    )
    assert p1.model is p2.model            # same key => one loaded instance, shared
    assert p1.model is not p3.model        # different model => its own instance
    assert len(built) == 2                 # not 3 — the dup never re-hit the factory


def test_node_pipeline_threads_the_run_device_pool(monkeypatch):
    # A per-node model override must reach the SAME device_pool the shared global pipeline
    # uses — was hardcoded to None, so multi-variant per-node builds never got memory-aware
    # allocation/eviction even when device_pool: was configured.
    captured = _capture_effective_model(monkeypatch)
    pool = object()
    _build_node_pipeline(
        _state(device_pool=pool), "text_embedding", {"model": "labse"},
    )
    assert captured["pool"] is pool


def test_node_pipeline_defaults_device_pool_to_none_when_absent(monkeypatch):
    # Duck-typed callers (tests, older state shapes) without a device_pool attribute at all
    # must not crash — getattr fallback, not s.device_pool direct access.
    captured = _capture_effective_model(monkeypatch)
    s = SimpleNamespace(
        config=EvaluationConfig(), cache_manager=None,
        current_node=SimpleNamespace(id="n"), service_provider=None,
    )  # no device_pool attribute at all
    _build_node_pipeline(s, "text_embedding", {"model": "labse"})
    assert captured["pool"] is None


def test_node_pipeline_noop_without_model_or_name(monkeypatch):
    # No per-node model => the global pipeline is used (no build) — parity for default runs.
    built = {"called": False}
    monkeypatch.setattr(
        "evaluator.evaluation.executor.node_pipeline._build_node_pipeline",
        lambda *a, **k: built.__setitem__("called", True),
    )
    s = _state()
    s.audio_embedding_pipeline = "GLOBAL"
    with _node_pipeline(s, "audio_embedding", {"k": 5}):  # no model/name
        assert s.audio_embedding_pipeline == "GLOBAL"
    assert built["called"] is False
