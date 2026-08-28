"""G5: the report is the single source for traces/judge (no top-level duplicates)."""

from evaluator.evaluation.handlers.rag import (
    _attach_traces_to_report,
    drop_mirrored_top_level_keys,
)


class _State:
    """Minimal RunState stand-in: a tiny (producer_id, artifact) -> value bus + the current
    node's bindings — enough for `_attach_traces_to_report`'s bus/sibling reads (R4)."""

    def __init__(self, bindings=(), bus=None):
        self._bindings = bindings
        self._bus = bus or {}

    def get_artifact(self, name, default=None):
        for art, pid in self._bindings:
            if art == name and (pid, name) in self._bus:
                return self._bus[(pid, name)]
        return default

    def _producers(self, name):
        return [pid for art, pid in self._bindings if art == name]

    def sibling_artifact(self, bound_name, extra_key, default=None):
        for pid in reversed(self._producers(bound_name)):
            if (pid, extra_key) in self._bus:
                return self._bus[(pid, extra_key)]
        return default


def test_attach_then_drop_consolidates_into_report():
    # retrieval_failure_analysis rides along as a top-level key of the bound `metrics`
    # artifact itself — already on own_report by the time finalize builds it.
    own_report = {"report": {}, "retrieval_failure_analysis": {"n_failed": 2}}
    s = _State(
        bindings=(
            ("query_traces", "build_query_traces"),
            ("generated_answers", "answer_gen"),
            ("judge_pass", "answer_judge"),
        ),
        bus={
            ("build_query_traces", "query_traces"): [{"query_id": "q1"}],
            ("answer_gen", "answer_generation"): {"cases": 5},
            ("answer_judge", "judge_summary"): {"llm_judge": {"cases": 3}},
        },
    )
    _attach_traces_to_report(s, own_report)
    assert own_report["report"]["traces"]["query_traces"] == [{"query_id": "q1"}]
    assert own_report["report"]["traces"]["answer_generation"] == {"cases": 5}
    assert own_report["report"]["traces"]["retrieval_failure_analysis"] == {"n_failed": 2}
    assert own_report["report"]["judge"] == {"llm_judge": {"cases": 3}}

    # At the output boundary the top-level duplicate is dropped (report is canonical).
    drop_mirrored_top_level_keys(own_report)
    assert "retrieval_failure_analysis" not in own_report
    assert own_report["report"]["traces"]["query_traces"] == [{"query_id": "q1"}]


def test_drop_is_noop_without_report():
    results = {"query_traces": [{"query_id": "q1"}]}  # no report assembled
    drop_mirrored_top_level_keys(results)
    assert results["query_traces"] == [{"query_id": "q1"}]  # not lost
