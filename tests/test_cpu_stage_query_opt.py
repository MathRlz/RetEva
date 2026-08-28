"""4b: query optimization's per-item rewrite/HyDE runs through `parallel_map`, so the
`cpu_stage_executor` knob (sync/thread/process) applies — every backend gives a byte-identical,
order-preserving result (sync == the serial loop), and a failing query falls back to its original.

Uses module-level fake optimizers (picklable for the process backend) in place of the real LLM
calls — the test exercises the plumbing + fallback, not the LLM."""
import functools

from evaluator.evaluation.executor.cpu_parallel import parallel_map
from evaluator.evaluation.handlers.query import _optimize_one_text


def _fake_opt(q, cfg):
    return f"OPT:{q}"


def _fake_opt_with_failure(q, cfg):
    if "bad" in q:
        raise ValueError("simulated optimizer failure")
    return f"OPT:{q}"


def _run(backend, fn):
    texts = [f"query {i}" if i % 3 else "bad query" for i in range(12)]
    unit = functools.partial(_optimize_one_text, fn=fn, cfg=None)
    return parallel_map(unit, texts, backend=backend, workers=2)


def test_query_opt_backends_are_byte_identical():
    # the unit returns (text, ok) so the caller can count fallbacks across any backend
    base = _run("sync", _fake_opt)
    assert _run("thread", _fake_opt) == base
    assert _run("process", _fake_opt) == base       # picklable fn + unit
    assert base == [
        (f"OPT:query {i}" if i % 3 else "OPT:bad query", True) for i in range(12)
    ]


def test_query_opt_falls_back_on_failure_and_reports_it():
    # The "bad query" items raise inside the optimizer → fall back to the original, in order,
    # each flagged ok=False so the stage can surface the count in provenance.
    out = _run("sync", _fake_opt_with_failure)
    assert out == [
        (f"OPT:query {i}", True) if i % 3 else ("bad query", False) for i in range(12)
    ]
    assert sum(1 for _text, ok in out if not ok) == 4
    # and the fallback is byte-identical across backends too
    assert _run("process", _fake_opt_with_failure) == out
