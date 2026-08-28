"""J1: the multi-aspect LLM judge — per-trace scoring, prompt assembly, aggregation. Uses a stub
LLM client (no real calls)."""

import json

from evaluator.config.judge import JudgeConfig
from evaluator.judge.core import build_judge_prompt, judge_trace, run_llm_judging


class _StubClient:
    """Returns a fixed JSON verdict for every call; records the prompts it saw."""

    def __init__(self, verdict):
        self._verdict = verdict
        self.prompts = []

    def call(self, messages, *, use_cache=False):
        self.prompts.append(messages[-1]["content"])
        return json.dumps(self._verdict)


_TRACE = {
    "query_id": "q1",
    "question": "What causes anemia?",
    "retrieved": [
        {"doc_key": "d1", "score": 0.9, "text": "Iron deficiency causes anemia."},
        {"doc_key": "d2", "score": 0.5, "text": "Unrelated text."},
    ],
    "generated_answer": "Iron deficiency.",
    "reference_answer": "Iron deficiency anemia.",
}


def _cfg(**over):
    base = dict(enabled=True, judge_aspects=["relevance", "faithfulness"], judge_mode="both")
    base.update(over)
    return JudgeConfig(**base)


def test_judge_trace_multi_aspect():
    client = _StubClient({"aspect_scores": {"relevance": 0.8, "faithfulness": 1.0},
                          "overall": 0.9, "verdict": "PASS", "reason": "good"})
    v = judge_trace(_TRACE, client=client, config=_cfg())
    assert v["aspect_scores"] == {"relevance": 0.8, "faithfulness": 1.0}
    assert v["overall"] == 0.9  # average of 0.8, 1.0
    assert v["verdict"] == "PASS" and v["reason"] == "good"


def test_overall_aggregation_weighted():
    client = _StubClient({"aspect_scores": {"relevance": 1.0, "faithfulness": 0.0},
                          "verdict": "PASS", "reason": ""})
    cfg = _cfg(score_aggregation="weighted",
               aspect_weights={"relevance": 0.75, "faithfulness": 0.25})
    v = judge_trace(_TRACE, client=client, config=cfg)
    assert abs(v["overall"] - 0.75) < 1e-9  # 1.0*0.75 + 0.0*0.25


def test_verdict_defaults_from_threshold():
    # LLM omits the verdict → derived from overall vs pass_threshold
    client = _StubClient({"aspect_scores": {"relevance": 0.3, "faithfulness": 0.3}, "reason": ""})
    v = judge_trace(_TRACE, client=client, config=_cfg(pass_threshold=0.5))
    assert v["overall"] == 0.3 and v["verdict"] == "FAIL"


def test_prompt_free_vs_graded_and_doc_text():
    free = build_judge_prompt(_TRACE, aspects=["relevance"], judge_mode="both",
                              reference_mode="free", include_doc_text=True, top_k=5)[1]
    graded = build_judge_prompt(_TRACE, aspects=["relevance"], judge_mode="both",
                                reference_mode="graded", include_doc_text=True, top_k=5)[1]
    assert "Iron deficiency causes anemia." in free  # doc text rendered
    assert "Reference answer" not in free            # free mode hides the reference
    assert "Reference answer" in graded              # graded mode shows it
    assert "relevance:" in free                      # aspect rubric line


def test_judge_metrics_registered_and_reference_free():
    from evaluator.evaluation.metric_registry import get_metric, list_metrics

    names = {m.name for m in list_metrics()}
    assert {"judge_overall", "judge_pass_rate", "judge_relevance", "judge_faithfulness"} <= names
    for name in ("judge_overall", "judge_relevance"):
        spec = get_metric(name)
        assert spec.gt is None  # reference-free → inputs == (scored,)


def test_judge_metrics_autoinject_only_configured_aspects():
    from evaluator.evaluation.metric_registry import applicable_metrics

    # judge ran with aspects=["relevance"] → only judge_overall/pass/relevance present
    available = {"judge_scores", "judge_pass", "judge_aspect_relevance"}
    fired = {m.name for m in applicable_metrics(available)
             if m.name.startswith("judge")}
    assert fired == {"judge_overall", "judge_pass_rate", "judge_relevance"}


def test_judge_overall_metric_computes():
    from evaluator.evaluation.item_set import ItemSet
    from evaluator.evaluation.metric_registry import compute_metric, get_metric

    scores = compute_metric(get_metric("judge_overall"),
                            {"judge_scores": ItemSet(["q1", "q2"], [0.8, 0.6])})
    assert list(scores) == [("q1", 0.8), ("q2", 0.6)]


def test_attach_judge_metrics_merges_into_report():
    # J3: the finalize-node merge scores the judge ItemSets via the registry and folds the
    # scalars into the existing report branch + the flat leaderboard keys.
    from types import SimpleNamespace

    from evaluator.evaluation.run_context import RunContext
    from evaluator.evaluation.item_set import ItemSet
    from evaluator.evaluation.handlers.metrics import attach_judge_metrics

    ctx = RunContext()
    ctx.put("answer_judge", "judge_scores", ItemSet(["q1", "q2"], [0.8, 0.6]))
    ctx.put("answer_judge", "judge_pass", ItemSet(["q1", "q2"], [1.0, 0.0]))
    ctx.put("answer_judge", "judge_aspect_relevance", ItemSet(["q1", "q2"], [0.9, 0.5]))

    results = {
        "report": {"branches": {"asr_text_retrieval": {"recall@5": {"mean": 0.7, "n": 2}}}},
        "asr_text_retrieval/recall@5": 0.7,
    }
    # R3-P1: the merge reads the finalize node's DECLARED judge bindings, not a bus scan.
    finalize = SimpleNamespace(id="finalize", bindings=(
        ("judge_scores", "answer_judge"), ("judge_pass", "answer_judge"),
        ("judge_aspect_relevance", "answer_judge"),
    ))
    s = SimpleNamespace(
        results=results, ctx=ctx, mode="asr_text_retrieval", current_node=finalize,
        metric_allowlist=None,
    )
    attach_judge_metrics(s, results)

    branch = results["report"]["branches"]["asr_text_retrieval"]
    assert abs(branch["judge_overall"]["mean"] - 0.7) < 1e-9       # (0.8 + 0.6) / 2
    assert abs(branch["judge_pass_rate"]["mean"] - 0.5) < 1e-9     # (1 + 0) / 2
    assert abs(branch["judge_relevance"]["mean"] - 0.7) < 1e-9     # (0.9 + 0.5) / 2
    assert branch["recall@5"]["mean"] == 0.7                       # pre-existing metric kept
    # flat leaderboard keys regenerated to include the judge metrics (+ the IR key preserved)
    assert abs(results["asr_text_retrieval/judge_overall"] - 0.7) < 1e-9
    assert abs(results["asr_text_retrieval/recall@5"] - 0.7) < 1e-9


def test_attach_judge_metrics_noop_without_judge():
    # No judge ItemSets on the bus (judge off — the parity default) → report untouched.
    from types import SimpleNamespace

    from evaluator.evaluation.run_context import RunContext
    from evaluator.evaluation.handlers.metrics import attach_judge_metrics

    results = {"report": {"branches": {"m": {"recall@5": {"mean": 0.7, "n": 2}}}}}
    finalize = SimpleNamespace(id="finalize", bindings=(("judge_scores", "answer_judge"),))
    s = SimpleNamespace(results=results, ctx=RunContext(), mode="m", current_node=finalize,
                        metric_allowlist=None)
    attach_judge_metrics(s, results)
    assert results["report"]["branches"]["m"] == {"recall@5": {"mean": 0.7, "n": 2}}


def test_publish_then_attach_end_to_end():
    # J6 wiring: the answer_judge publish half (_publish_judge_scores) and the finalize merge
    # half (attach_judge_metrics) agree on artifact names + shapes — judge details in → judge
    # metrics on the report out, through the real handler code, no LLM/models needed.
    from types import SimpleNamespace

    from evaluator.evaluation.run_context import RunContext
    from evaluator.evaluation.handlers.rag import _publish_judge_scores
    from evaluator.evaluation.handlers.metrics import attach_judge_metrics

    ctx = RunContext()

    class _State:  # minimal stand-in: put_items + the fields the two helpers read
        def __init__(self):
            self.ctx = ctx
            self.current_node = SimpleNamespace(id="answer_judge")
            self.judge_config = SimpleNamespace(judge_aspects=["relevance", "faithfulness"])
            self.mode = "asr_text_retrieval"
            self.metric_allowlist = None
            self.results = {
                "report": {"branches": {self.mode: {"recall@5": {"mean": 0.5, "n": 2}}}},
            }

        def put_items(self, name, items):  # identical to RunState.put_items
            self.ctx.put(self.current_node.id, name, items)

    s = _State()
    details = [
        {"query_id": "q1", "judge": {"overall": 0.8, "verdict": "PASS",
                                     "aspect_scores": {"relevance": 0.9, "faithfulness": 0.7}}},
        {"query_id": "q2", "judge": {"overall": 0.4, "verdict": "FAIL",
                                     "aspect_scores": {"relevance": 0.5, "faithfulness": 0.3}}},
    ]
    _publish_judge_scores(s, details)
    # The executor swaps current_node per stage: the merge runs on the finalize node,
    # whose declared bindings carry the judge artifacts (R3-P1).
    s.current_node = SimpleNamespace(id="finalize", bindings=(
        ("judge_scores", "answer_judge"), ("judge_pass", "answer_judge"),
        ("judge_aspect_relevance", "answer_judge"),
        ("judge_aspect_faithfulness", "answer_judge"),
    ))
    attach_judge_metrics(s, s.results)

    branch = s.results["report"]["branches"]["asr_text_retrieval"]
    assert abs(branch["judge_overall"]["mean"] - 0.6) < 1e-9      # (0.8 + 0.4) / 2
    assert abs(branch["judge_pass_rate"]["mean"] - 0.5) < 1e-9    # PASS, FAIL
    assert abs(branch["judge_relevance"]["mean"] - 0.7) < 1e-9    # (0.9 + 0.5) / 2
    assert abs(branch["judge_faithfulness"]["mean"] - 0.5) < 1e-9  # (0.7 + 0.3) / 2
    assert abs(s.results["asr_text_retrieval/judge_overall"] - 0.6) < 1e-9  # flat leaderboard key


def test_run_llm_judging_aggregates(monkeypatch):
    import evaluator.judge.core as jc

    monkeypatch.setattr(jc, "judge_trace", lambda t, **kw: {
        "overall": 0.6 if t["query_id"] == "a" else 0.8,
        "aspect_scores": {"relevance": 0.6 if t["query_id"] == "a" else 0.8},
        "verdict": "FAIL" if t["query_id"] == "a" else "PASS", "reason": "",
    })
    out = run_llm_judging([{"query_id": "a"}, {"query_id": "b"}], _cfg(judge_aspects=["relevance"]))
    assert out["cases"] == 2
    assert abs(out["mean_score"] - 0.7) < 1e-9
    assert abs(out["aspect_means"]["relevance"] - 0.7) < 1e-9
    assert out["pass_rate"] == 0.5
