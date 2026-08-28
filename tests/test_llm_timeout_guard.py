"""A serial-sized timeout plus concurrency is a timeout storm — catch it at validate time.

Observed: concurrency 4 with timeout_s 120 on a full-text run produced continuous
`LLM call failed ... timed out`, each one retried twice, which added load to the very server
that was already saturated. The numbers are knowable before the run starts.
"""

from evaluator.config.evaluation import EvaluationConfig


def _warnings(**llm):
    cfg = EvaluationConfig.from_dict({
        "llm": llm,
        "judge": {"enabled": True},
        "answer_generation": {"enabled": True},
    })
    return [w for w in cfg.validate() if "concurrency" in w]


def test_warns_when_the_timeout_was_sized_for_serial_calls():
    warned = _warnings(concurrency=4, timeout_s=120)
    assert len(warned) == 2                       # answer generation AND judge
    assert "timeout_s >= 240" in warned[0]        # says what to set it to


def test_quiet_when_the_timeout_scales_with_concurrency():
    assert _warnings(concurrency=4, timeout_s=600) == []


def test_quiet_for_serial_runs():
    assert _warnings(concurrency=1, timeout_s=60) == []


def test_disabled_components_are_not_warned_about():
    cfg = EvaluationConfig.from_dict({"llm": {"concurrency": 8, "timeout_s": 30}})
    assert [w for w in cfg.validate() if "concurrency" in w] == []
