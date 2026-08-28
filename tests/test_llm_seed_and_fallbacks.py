"""P4 pins: LLM sampling seed rides the request; optimization fallbacks reach provenance."""

import json
import urllib.request

from evaluator.config.llm_backend import LLMConfig
from evaluator.llm_client.client import LLMClient
from evaluator.evaluation.provenance import build_provenance


def _sent_body(monkeypatch, cfg):
    """Capture the JSON body the client would POST."""
    captured = {}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps(
                {"choices": [{"message": {"content": "ok"}}]}
            ).encode()

    def fake_urlopen(req, timeout=None):
        captured["body"] = json.loads(req.data.decode())
        return _Resp()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    LLMClient(cfg).call([{"role": "user", "content": "hi"}])
    return captured["body"]


def test_seed_is_sent_when_configured(monkeypatch):
    body = _sent_body(monkeypatch, LLMConfig(seed=1234))
    assert body["seed"] == 1234


def test_seed_field_absent_by_default(monkeypatch):
    body = _sent_body(monkeypatch, LLMConfig())
    assert "seed" not in body  # unknown keys upset some servers


def test_optimization_fallbacks_surface_in_provenance():
    prov = build_provenance(None, optimization_fallbacks=3)
    assert prov["optimization_fallbacks"] == 3
    # absent when every item optimized — a clean run's report is unchanged
    assert "optimization_fallbacks" not in build_provenance(None, optimization_fallbacks=0)
    assert "optimization_fallbacks" not in build_provenance(None)
