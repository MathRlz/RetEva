"""TTS providers now actually move their model to the configured device (m4t/mms:
transformers `.to(device)`; xtts_v2: Coqui's own `.to(device)`) — before this they always
ran on CPU regardless of `config.device`. Piper (an external CPU-only subprocess binary)
has no device concept and keeps the base no-op `.to()`.
"""
from dataclasses import dataclass
from typing import Optional

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


@dataclass
class _Cfg:
    voice: Optional[str] = None
    language: str = "en"
    device: str = "cpu"


class _FakeHFModel:
    """Stands in for a transformers model: records every `.to(device)` call."""

    def __init__(self):
        self.devices_seen = []
        self.config = type("C", (), {"sampling_rate": 16000})()

    def to(self, device):
        self.devices_seen.append(str(device))
        return self

    def eval(self):
        return self

    def generate(self, **kwargs):
        return torch.zeros(1, 100)

    def __call__(self, **kwargs):
        return type("Out", (), {"waveform": torch.zeros(1, 100)})()


class _FakeProcessorOutput(dict):
    def to(self, device):
        self._device = device
        return self


class _FakeProcessor:
    def __call__(self, *a, **k):
        return _FakeProcessorOutput()

    @classmethod
    def from_pretrained(cls, *a, **k):
        return cls()


def test_m4t_tts_moves_model_to_configured_device(monkeypatch):
    # transformers' top-level namespace is a _LazyModule (custom __getattr__) — patching a
    # top-level name there is unreliable across import paths. Patch from_pretrained on the
    # already-resolved class instead: robust regardless of how the lazy module caches names.
    from transformers import AutoProcessor, SeamlessM4Tv2ForTextToSpeech

    fake_model = _FakeHFModel()
    monkeypatch.setattr(AutoProcessor, "from_pretrained", classmethod(lambda cls, *a, **k: _FakeProcessor()))
    monkeypatch.setattr(
        SeamlessM4Tv2ForTextToSpeech, "from_pretrained", classmethod(lambda cls, *a, **k: fake_model),
    )
    from evaluator.models.tts.m4t_tts import M4TTTS

    provider = M4TTTS(_Cfg(device="cuda:0"))
    assert fake_model.devices_seen == ["cuda:0"]  # moved at construction

    provider.to("cuda:1")
    assert fake_model.devices_seen == ["cuda:0", "cuda:1"]
    assert provider.device == "cuda:1"


def test_mms_tts_moves_model_to_configured_device(monkeypatch):
    from transformers import AutoTokenizer, VitsModel

    fake_model = _FakeHFModel()
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", classmethod(lambda cls, *a, **k: _FakeProcessor()))
    monkeypatch.setattr(VitsModel, "from_pretrained", classmethod(lambda cls, *a, **k: fake_model))
    from evaluator.models.tts.mms_tts import MMSTTS

    provider = MMSTTS(_Cfg(device="cuda:0"))
    assert fake_model.devices_seen == ["cuda:0"]

    provider.to("cpu")
    assert fake_model.devices_seen == ["cuda:0", "cpu"]
    assert provider.device == "cpu"


def test_mms_tts_defaults_to_cpu_when_device_unset(monkeypatch):
    from transformers import AutoTokenizer, VitsModel

    fake_model = _FakeHFModel()
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", classmethod(lambda cls, *a, **k: _FakeProcessor()))
    monkeypatch.setattr(VitsModel, "from_pretrained", classmethod(lambda cls, *a, **k: fake_model))
    from evaluator.models.tts.mms_tts import MMSTTS

    MMSTTS(_Cfg())  # no device set -> AudioSynthesisConfig-style default "cpu"
    assert fake_model.devices_seen == ["cpu"]


def test_piper_tts_to_is_a_no_op(monkeypatch):
    from evaluator.models.tts.base_tts import BaseTTSModel

    model = BaseTTSModel(_Cfg())
    assert model.to("cuda:0") is model  # no-op, returns self, doesn't raise
