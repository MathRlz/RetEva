"""B4/F18: decompose_query + generate_multi_queries share one _fanout / _parse_listed_queries.

No LLM is called — `_call_llm` is monkeypatched. Pins the parse cleanup + the two fan-out
contracts (decompose returns the parsed list as-is; multi_query prepends the de-duplicated
original) and the disabled / empty / error fallbacks to ``[query]``.
"""

from evaluator.config import QueryOptimizationConfig
from evaluator.models.retrieval.query import optimization as opt


def _cfg(method):
    return QueryOptimizationConfig(enabled=True, method=method)


def test_parse_listed_queries_cleans_numbering_comments_blanks():
    raw = "# header\n1. first query\n2) second query\n- third\n\n   \n4. "
    assert opt._parse_listed_queries(raw) == ["first query", "second query", "third"]


def test_decompose_returns_parsed_list(monkeypatch):
    monkeypatch.setattr(opt, "_call_llm", lambda *a, **k: "1. symptoms\n2. treatment")
    assert opt.decompose_query("diabetes?", _cfg("decompose")) == ["symptoms", "treatment"]


def test_multi_query_prepends_deduped_original(monkeypatch):
    # response repeats the original (different case) — must not duplicate it
    monkeypatch.setattr(
        opt, "_call_llm", lambda *a, **k: "heart attack symptoms\nMI signs"
    )
    out = opt.generate_multi_queries("heart attack symptoms", _cfg("multi_query"))
    assert out == ["heart attack symptoms", "MI signs"]


def test_disabled_or_wrong_method_returns_original():
    assert opt.decompose_query("q", QueryOptimizationConfig(enabled=False)) == ["q"]
    assert opt.generate_multi_queries("q", _cfg("decompose")) == ["q"]


def test_empty_and_error_fall_back_to_original(monkeypatch):
    monkeypatch.setattr(opt, "_call_llm", lambda *a, **k: "# only comments\n\n")
    assert opt.decompose_query("q", _cfg("decompose")) == ["q"]

    def _boom(*a, **k):
        raise RuntimeError("llm down")

    monkeypatch.setattr(opt, "_call_llm", _boom)
    assert opt.generate_multi_queries("q", _cfg("multi_query")) == ["q"]
