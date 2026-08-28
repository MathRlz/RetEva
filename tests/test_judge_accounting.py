"""Every attempted judgement is accounted for, and a truncated reply gets one more chance.

A failed case used to vanish: logged once, filtered out, and `cases` (successes only) was the
only number in the report — so "judged 900 of 1209" read as "cases: 900" with no trace of the
309 that were lost.
"""

import pytest

from evaluator.config.judge import JudgeConfig
from evaluator.judge import core


def _traces(n=4):
    return [{"query_id": f"q{i}", "question": f"q{i}?", "generated_answer": "a",
             "retrieved": []} for i in range(n)]


def _cfg(**kw):
    return JudgeConfig(enabled=True, model="m", use_local_server=True,
                       local_server_url="http://x/v1/chat/completions", **kw)


_GOOD = '{"aspect_scores": {"relevance": 0.8}, "overall": 0.8, "verdict": "PASS", "reason": "ok"}'
_TRUNCATED = '{"aspect_scores": {'


class _Client:
    """Replies from a script; records the system prompts it was given."""

    def __init__(self, replies):
        self._replies = list(replies)
        self.system_prompts = []

    def call(self, messages, use_cache=False):
        self.system_prompts.append(messages[0]["content"])
        reply = self._replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def _run(monkeypatch, client, traces, cfg):
    monkeypatch.setattr(core, "LLMClient", lambda *a, **kw: client)
    return core.run_llm_judging(traces, cfg)


def test_max_cases_zero_judges_every_trace(monkeypatch):
    client = _Client([_GOOD] * 4)
    out = _run(monkeypatch, client, _traces(4), _cfg(max_cases=0))
    assert out["cases"] == 4 and out["attempted"] == 4


def test_max_cases_positive_still_caps(monkeypatch):
    client = _Client([_GOOD] * 2)
    out = _run(monkeypatch, client, _traces(4), _cfg(max_cases=2))
    assert out["cases"] == 2 and out["attempted"] == 2


def test_truncated_reply_is_re_asked_once_and_recovered(monkeypatch):
    client = _Client([_TRUNCATED, _GOOD])
    out = _run(monkeypatch, client, _traces(1), _cfg(max_cases=0))
    assert out["cases"] == 1 and out["failed_cases"] == 0
    assert "JSON ONLY" in client.system_prompts[1]      # the retry asks for less prose
    assert "JSON ONLY" not in client.system_prompts[0]


def test_permanent_failure_is_counted_not_hidden(monkeypatch):
    # every case fails twice (first reply + retry)
    client = _Client([_TRUNCATED, _TRUNCATED, _GOOD, _GOOD])
    out = _run(monkeypatch, client, _traces(2), _cfg(max_cases=0))
    assert out["cases"] == 1
    assert out["failed_cases"] == 1
    assert out["attempted"] == 2
    assert out["cases"] + out["failed_cases"] == out["attempted"]
    assert sum(out["failures"].values()) == 1          # keyed by exception type


def test_a_bad_case_never_aborts_the_stage(monkeypatch):
    client = _Client([RuntimeError("boom"), _GOOD, _GOOD, _GOOD])
    out = _run(monkeypatch, client, _traces(3), _cfg(max_cases=0))
    assert out["cases"] == 2 and out["failed_cases"] == 1
    assert [d["query_id"] for d in out["details"]] == ["q1", "q2"]


@pytest.mark.parametrize("n", [1, 5])
def test_accounting_invariant_holds(monkeypatch, n):
    client = _Client([_GOOD] * n)
    out = _run(monkeypatch, client, _traces(n), _cfg(max_cases=0))
    assert out["cases"] + out["failed_cases"] == out["attempted"] == n
