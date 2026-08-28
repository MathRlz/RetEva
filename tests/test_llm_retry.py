"""A4-remainder pins (CRITIQUE.md): transient LLM failures retry; permanent ones don't.

One network hiccup used to drop the item (drop-and-log shrinks the eval set and biases
paired deltas); a 401 must still fail immediately — retrying auth errors can't help.
"""

import json
import urllib.error
import urllib.request

import pytest

from evaluator.config.llm_backend import LLMConfig
from evaluator.llm_client.client import LLMClient

_MESSAGES = [{"role": "user", "content": "hi"}]


class _Resp:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode()


def _client(retries=2):
    return LLMClient(LLMConfig(api_key_env="X_NO_KEY", max_retries=retries))


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)


def test_transient_failure_retries_and_succeeds(monkeypatch):
    calls = []

    def fake_urlopen(req, timeout=None):
        calls.append(1)
        if len(calls) == 1:
            raise urllib.error.URLError("connection timed out")
        return _Resp()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    assert _client().call(_MESSAGES) == "ok"
    assert len(calls) == 2  # one failure + one successful retry, item survives


def test_auth_error_fails_immediately(monkeypatch):
    calls = []

    def fake_urlopen(req, timeout=None):
        calls.append(1)
        raise urllib.error.HTTPError(req.full_url, 401, "unauthorized", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(RuntimeError, match="LLM call failed"):
        _client().call(_MESSAGES)
    assert len(calls) == 1  # no retry on 401


def test_persistent_5xx_exhausts_retries(monkeypatch):
    calls = []

    def fake_urlopen(req, timeout=None):
        calls.append(1)
        raise urllib.error.HTTPError(req.full_url, 503, "unavailable", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(RuntimeError, match="LLM call failed"):
        _client(retries=2).call(_MESSAGES)
    assert len(calls) == 3  # initial + 2 retries


def test_zero_retries_disables_the_loop(monkeypatch):
    calls = []

    def fake_urlopen(req, timeout=None):
        calls.append(1)
        raise urllib.error.URLError("timed out")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(RuntimeError):
        _client(retries=0).call(_MESSAGES)
    assert len(calls) == 1
