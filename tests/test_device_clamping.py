"""Out-of-range `cuda:N` (CUDA_VISIBLE_DEVICES narrowed the visible set, or the config just
assumes more GPUs than this launch has) must not crash with "invalid device ordinal" —
`_clamp_to_available_device` remaps it to an ordinal that actually exists instead."""

import sys
from types import SimpleNamespace

import pytest

from evaluator.pipeline import factory as factory_mod


def _install_fake_torch(monkeypatch, device_count):
    fake = SimpleNamespace(cuda=SimpleNamespace(
        is_available=lambda: device_count > 0,
        device_count=lambda: device_count,
    ))
    monkeypatch.setitem(sys.modules, "torch", fake)
    factory_mod._WARNED_DEVICE_CLAMPS.clear()


@pytest.mark.parametrize("count,requested,expected", [
    (1, "cuda:1", "cuda:0"),   # the reported crash: config wants cuda:1, only 1 GPU visible
    (1, "cuda:0", "cuda:0"),   # in range — unchanged
    (2, "cuda:1", "cuda:1"),   # in range — unchanged
    (2, "cuda:3", "cuda:1"),   # out of range on a 2-GPU box — cyclic remap, not always 0
    (0, "cuda:0", "cpu"),      # no CUDA at all — fall back to cpu
    (3, "cpu", "cpu"),         # non-cuda strings pass through untouched
    (3, "cuda", "cuda"),       # bare "cuda" (ordinal-less) passes through untouched
])
def test_clamp_to_available_device(monkeypatch, count, requested, expected):
    _install_fake_torch(monkeypatch, count)
    assert factory_mod._clamp_to_available_device(requested) == expected


def test_clamp_warns_once_per_distinct_device(monkeypatch, caplog):
    _install_fake_torch(monkeypatch, 1)
    import logging
    with caplog.at_level(logging.WARNING):
        factory_mod._clamp_to_available_device("cuda:1")
        factory_mod._clamp_to_available_device("cuda:1")
        factory_mod._clamp_to_available_device("cuda:2")
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 2  # cuda:1 warned once (not twice), cuda:2 warned once


def test_model_builders_clamps_when_no_device_pool(monkeypatch):
    _install_fake_torch(monkeypatch, 1)
    mcfg = SimpleNamespace()
    builders = factory_mod._ModelBuilders(SimpleNamespace(model=mcfg), service_provider=None,
                                          device_pool=None)
    assert builders._get_device("text_embedding", "labse", "cuda:1") == "cuda:0"


def test_model_builders_leaves_device_pool_untouched(monkeypatch):
    # A configured device_pool already bounds itself to real GPUs — _get_device must defer
    # to it entirely (not double-clamp its result).
    _install_fake_torch(monkeypatch, 1)
    mcfg = SimpleNamespace()
    pool = SimpleNamespace(allocate=lambda category, memory_gb: "cuda:7")
    builders = factory_mod._ModelBuilders(SimpleNamespace(model=mcfg), service_provider=None,
                                          device_pool=pool)
    monkeypatch.setattr(factory_mod, "estimate_model_memory_gb", lambda *a, **k: 1.0, raising=False)
    import evaluator.config as config_mod
    monkeypatch.setattr(config_mod, "estimate_model_memory_gb", lambda *a, **k: 1.0)
    assert builders._get_device("text_embedding", "labse", "cuda:1") == "cuda:7"
