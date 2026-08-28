"""A failed LLM call must say what the SERVER said.

Observed: four concurrent requests to a local server whose slots each reserved the model's full
128k context; the server 500'd on KV allocation, and the client reported only "timed out" /
"HTTP Error 500" — nothing pointing at OLLAMA_CONTEXT_LENGTH, which was the actual fix.
"""

import io
import urllib.error

import pytest

from evaluator.config.llm_backend import LLMConfig
from evaluator.llm_client.client import LLMClient


def _client(**kw):
    cfg = LLMConfig(model="m", use_local_server=True,
                    local_server_url="http://localhost:11434/v1/chat/completions",
                    max_retries=0, **kw)
    return LLMClient(cfg, component="test"), cfg


def _raise_http_500(*_a, **_kw):
    raise urllib.error.HTTPError(
        "http://localhost:11434/v1/chat/completions", 500, "Internal Server Error", {},
        io.BytesIO(b'{"error":"failed to allocate KV cache: out of memory"}'),
    )


def test_server_error_body_reaches_the_message(monkeypatch):
    client, _ = _client()
    monkeypatch.setattr("urllib.request.urlopen", _raise_http_500)
    with pytest.raises(RuntimeError, match="failed to allocate KV cache"):
        client.call([{"role": "user", "content": "hi"}])


def test_concurrent_runs_get_the_vram_hint(monkeypatch):
    client, _ = _client(concurrency=4)
    monkeypatch.setattr("urllib.request.urlopen", _raise_http_500)
    with pytest.raises(RuntimeError, match="OLLAMA_CONTEXT_LENGTH"):
        client.call([{"role": "user", "content": "hi"}])


def test_serial_runs_are_not_told_about_concurrency(monkeypatch):
    client, _ = _client(concurrency=1)
    monkeypatch.setattr("urllib.request.urlopen", _raise_http_500)
    with pytest.raises(RuntimeError) as exc:
        client.call([{"role": "user", "content": "hi"}])
    assert "concurrency=" not in str(exc.value)
