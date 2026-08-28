"""Auto device selection maps torch's device count onto the three model families.

Exclusion is torch's job now (``CUDA_VISIBLE_DEVICES`` / ``HIP_VISIBLE_DEVICES``), so this
mapping is all that stands between "N GPUs" and where the models land.
"""

import sys
from types import SimpleNamespace

import pytest

from evaluator.config.model import ModelConfig


def _install_fake_torch(monkeypatch, device_count):
    fake = SimpleNamespace(cuda=SimpleNamespace(
        is_available=lambda: device_count > 0,
        device_count=lambda: device_count,
    ))
    monkeypatch.setitem(sys.modules, "torch", fake)


@pytest.mark.parametrize("count,asr,text,audio", [
    (0, "cpu", "cpu", "cpu"),
    (1, "cuda:0", "cuda:0", "cuda:0"),
    (2, "cuda:0", "cuda:1", "cuda:0"),      # text embedding spreads to the second GPU
    (3, "cuda:0", "cuda:1", "cuda:0"),      # extra GPUs stay free
])
def test_auto_configure_devices(monkeypatch, count, asr, text, audio):
    _install_fake_torch(monkeypatch, count)
    cfg = ModelConfig()
    cfg.auto_configure_devices()
    assert (cfg.asr_device, cfg.text_emb_device, cfg.audio_emb_device) == (asr, text, audio)
