"""Runtime OOM recovery (T7b) + per-model pool keys: allocate dedup, LRU eviction on
minimal-batch OOM, ManualStrategy prefix matching, and _ModelBuilders pool-key identity."""

from types import SimpleNamespace

import pytest

from evaluator.devices import pool as pool_module
from evaluator.devices.memory import run_with_oom_backoff
from evaluator.devices.monitor import MemoryInfo
from evaluator.devices.pool import GPUPool
from evaluator.devices.strategy import ManualStrategy


class _FakeMonitor:
    def __init__(self, total_gb=100.0):
        self._total = total_gb

    def get_device_count(self):
        return 2

    def get_memory_usage(self, device_idx):
        return MemoryInfo(total=self._total, used=0.0, free=self._total)


def _pool(total_gb=100.0):
    return GPUPool(
        ["cuda:0", "cuda:1"], monitor=_FakeMonitor(total_gb),
        memory_buffer_percent=0.0, allow_cpu_fallback=True,
    )


def test_allocate_is_idempotent_per_key():
    # Two builds of the same model (same pool key) must not stack reservations.
    pool = _pool(total_gb=10.0)
    dev1 = pool.allocate("text_embedding:jina_v4", 6.0)
    dev2 = pool.allocate("text_embedding:jina_v4", 6.0)
    assert dev1 == dev2
    usage = pool.get_usage()[dev1]
    assert usage.reserved_memory_gb == 6.0  # once, not 12


def test_distinct_keys_get_distinct_reservations():
    pool = _pool(total_gb=10.0)
    pool.allocate("text_embedding:jina_v4", 6.0)
    pool.allocate("text_embedding:eam_alignment", 6.0)
    reserved = {d: u.reserved_memory_gb for d, u in pool.get_usage().items()}
    # 6+6 doesn't fit one 10GB GPU → they must land on different devices.
    assert sorted(v for v in reserved.values() if v) == [6.0, 6.0]
    assert pool.get_device_for_model("text_embedding:jina_v4") != \
        pool.get_device_for_model("text_embedding:eam_alignment")


def test_evict_lru_one_evicts_oldest_with_callback():
    pool = _pool()
    evicted = []
    pool.allocate("a", 1.0)
    pool.allocate("b", 1.0)
    pool.register_eviction_callback("a", lambda: evicted.append("a"))
    pool.register_eviction_callback("b", lambda: evicted.append("b"))
    pool.touch("a")  # b becomes LRU head... a moved to tail
    assert pool.evict_lru_one() is True
    assert evicted == ["b"]
    assert pool.get_device_for_model("b") is None
    assert pool.evict_lru_one() is True
    assert evicted == ["b", "a"]
    assert pool.evict_lru_one() is False  # nothing left


def test_minimal_batch_oom_evicts_and_retries(monkeypatch):
    # Backoff at batch=1: evict one parked model, retry succeeds.
    pool = _pool()
    pool.allocate("parked", 1.0)
    pool.register_eviction_callback("parked", lambda: None)
    monkeypatch.setattr(pool_module, "_active_pool", pool)

    calls = {"n": 0}

    def fn(items):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("CUDA out of memory. Tried to allocate 608.00 MiB")
        return [x * 2 for x in items]

    assert run_with_oom_backoff(fn, [21]) == [42]
    assert calls["n"] == 2
    assert pool.get_device_for_model("parked") is None  # was evicted


def test_minimal_batch_oom_reraises_when_nothing_evictable(monkeypatch):
    pool = _pool()  # no registered callbacks → nothing evictable
    monkeypatch.setattr(pool_module, "_active_pool", pool)

    def fn(items):
        raise RuntimeError("CUDA out of memory")

    with pytest.raises(RuntimeError, match="out of memory"):
        run_with_oom_backoff(fn, [1])


def test_minimal_batch_oom_reraises_without_active_pool(monkeypatch):
    monkeypatch.setattr(pool_module, "_active_pool", None)

    def fn(items):
        raise RuntimeError("CUDA out of memory")

    with pytest.raises(RuntimeError, match="out of memory"):
        run_with_oom_backoff(fn, [1])


def test_manual_strategy_matches_segmented_keys():
    strategy = ManualStrategy({
        "text_embedding": "cuda:1",
        "text_embedding:jina_v4": "cuda:0",
    })
    # Most-specific first: full key > category:type > category.
    assert strategy.allocate(None, "text_embedding:jina_v4:jinaai/x", 1.0) == "cuda:0"
    assert strategy.allocate(None, "text_embedding:eam_alignment:/ckpt/eam.pt", 1.0) == "cuda:1"
    assert strategy.allocate(None, "audio_embedding:attention_pool", 1.0) is None


def test_tts_synthesis_allocates_and_releases_pool_device(monkeypatch):
    # GPU-capable TTS provider goes through the pool: device comes from allocate(),
    # reservation is released when synthesis finishes (the model is freed there).
    from evaluator.config.audio_synthesis import AudioSynthesisConfig
    from evaluator.pipeline.audio import prepare as prepare_module
    from evaluator.pipeline.audio.prepare import synthesize_missing_query_audio

    pool = _pool(total_gb=20.0)
    seen_devices = []

    class _FakeSynth:
        def __init__(self, cfg):
            seen_devices.append(cfg.device)

        def synthesize(self, text, output_path=None):
            return None

    monkeypatch.setattr(
        "evaluator.pipeline.audio.synthesis.AudioSynthesizer", _FakeSynth
    )
    monkeypatch.setattr(prepare_module, "_release_torch_memory", lambda: None)

    q = SimpleNamespace(question_id="q1", question_text="hello", audio_path=None, language=None)
    cfg = AudioSynthesisConfig(provider="m4t", output_dir=None, device="cuda:0")

    # Occupy cuda:0 so the 10GB m4t reservation must land on cuda:1.
    pool.allocate("hog", 15.0)
    synthesize_missing_query_audio([q], cfg, device_pool=pool)

    assert seen_devices == ["cuda:1"]  # pool decision, not the config's cuda:0
    assert pool.get_device_for_model("tts:m4t") is None  # released after synthesis


def test_cpu_only_tts_provider_skips_pool():
    from evaluator.config import estimate_model_memory_gb

    assert estimate_model_memory_gb("tts", "piper") == 0.0
    assert estimate_model_memory_gb("tts", "m4t") > 0


def test_evict_lru_one_prefers_failing_device():
    pool = _pool()
    evicted = []
    pool.allocate("on_gpu0", 1.0)  # most-free tie → cuda:0
    pool.allocate("on_gpu1", 2.0)  # cuda:1 now freer? no — allocate picks most free ⇒ cuda:1
    pool.register_eviction_callback("on_gpu0", lambda: evicted.append("on_gpu0"))
    pool.register_eviction_callback("on_gpu1", lambda: evicted.append("on_gpu1"))
    dev1 = pool.get_device_for_model("on_gpu1")
    # Even though on_gpu0 is older in LRU, prefer_device targets the failing GPU first.
    assert pool.evict_lru_one(prefer_device=dev1) is True
    assert evicted == ["on_gpu1"]


def test_minimal_batch_oom_prefers_device_from_message(monkeypatch):
    pool = _pool()
    evicted = []
    pool.allocate("a", 1.0)
    pool.allocate("b", 1.0)
    pool.register_eviction_callback("a", lambda: evicted.append("a"))
    pool.register_eviction_callback("b", lambda: evicted.append("b"))
    dev_b = pool.get_device_for_model("b")
    monkeypatch.setattr(pool_module, "_active_pool", pool)

    calls = {"n": 0}

    def fn(items):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError(
                f"CUDA out of memory. Tried to allocate 608.00 MiB. "
                f"GPU {dev_b.split(':')[1]} has a total capacity of 23.55 GiB"
            )
        return list(items)

    assert run_with_oom_backoff(fn, [1]) == [1]
    assert evicted == ["b"]  # the model on the OOMing GPU, not the LRU head


def test_eam_encode_batches_internally():
    from evaluator.models.a2e.eam_alignment import EamAlignmentModel
    import numpy as _np
    import torch as _torch

    seen = []

    class _FakeCore:
        embedding_dim = 512

        def encode_text(self, texts):
            seen.append(len(texts))
            return _torch.zeros(len(texts), 512)

    stub = SimpleNamespace(core=_FakeCore(), TEXT_BATCH_SIZE=EamAlignmentModel.TEXT_BATCH_SIZE)
    out = EamAlignmentModel.encode(stub, [f"doc {i}" for i in range(70)])
    assert out.shape == (70, 512)
    assert seen == [32, 32, 6]  # chunked, not one 70-doc forward
    assert EamAlignmentModel.encode(stub, []).shape == (0, 512)


def test_on_use_soft_cpu_policy_is_valid():
    from evaluator.config.service_runtime import ServiceRuntimeConfig

    cfg = ServiceRuntimeConfig(offload_policy="on_use_soft_cpu")
    assert cfg.offload_policy == "on_use_soft_cpu"
    with pytest.raises(ValueError):
        ServiceRuntimeConfig(offload_policy="bogus")


def test_release_node_model_parks_and_returns_reservation():
    # Aggressive offload: provider soft-parks the model, pool reservation is returned.
    from evaluator.evaluation.executor.node_pipeline import _release_node_model

    pool = _pool()
    device = pool.allocate("text_embedding:jina_v4", 5.0)
    released = []

    class _Provider:
        def release_model_instance(self, model, *, soft_cpu=False):
            released.append((model, soft_cpu))
            return True

    model = SimpleNamespace(_gpu_pool_key="text_embedding:jina_v4")
    state = SimpleNamespace(
        service_provider=_Provider(), device_pool=pool,
        current_node=SimpleNamespace(id="corpus_embedding_apm_whisper"),
    )
    _release_node_model(state, SimpleNamespace(model=model))

    assert released == [(model, True)]
    assert pool.get_device_for_model("text_embedding:jina_v4") is None
    # Device is reusable immediately.
    assert pool.get_usage()[device].reserved_memory_gb == 0.0


def test_release_node_model_survives_provider_failure():
    from evaluator.evaluation.executor.node_pipeline import _release_node_model

    class _Provider:
        def release_model_instance(self, model, *, soft_cpu=False):
            raise RuntimeError("boom")

    state = SimpleNamespace(
        service_provider=_Provider(), device_pool=None,
        current_node=SimpleNamespace(id="n"),
    )
    # Must not raise — offload is best-effort.
    _release_node_model(state, SimpleNamespace(model=SimpleNamespace()))


def test_model_builders_pool_key_distinguishes_models():
    from evaluator.pipeline.factory import _ModelBuilders

    def builders(**model_fields):
        mcfg = SimpleNamespace(**model_fields)
        return _ModelBuilders(SimpleNamespace(model=mcfg), None, None)

    jina = builders(text_emb_model_name="jinaai/jina-embeddings-v4")
    eam = builders(text_emb_model_path="/ckpt/eam_alignment.pt")
    key_jina = jina._pool_key("text_embedding", "jina_v4")
    key_eam = eam._pool_key("text_embedding", "eam_alignment")
    assert key_jina != key_eam
    assert key_jina.startswith("text_embedding:jina_v4")
    assert key_eam.startswith("text_embedding:eam_alignment")

    # Same model resolved by two nodes → same key (shares one reservation).
    apm1 = builders(audio_emb_model_path="/ckpt/apm_whisper.pt", audio_emb_encoder_type="whisper")
    apm2 = builders(audio_emb_model_path="/ckpt/apm_whisper.pt", audio_emb_encoder_type="whisper")
    assert apm1._pool_key("audio_embedding", "attention_pool") == \
        apm2._pool_key("audio_embedding", "attention_pool")

    # Different checkpoint (apm_m4t) → different key.
    apm_m4t = builders(audio_emb_model_path="/ckpt/apm_m4t.pt", audio_emb_encoder_type="m4t")
    assert apm_m4t._pool_key("audio_embedding", "attention_pool") != \
        apm1._pool_key("audio_embedding", "attention_pool")
