"""M3: a single failing trace (LLM timeout/transient error) must not abort the whole judge
stage and lose the other verdicts — it is dropped and logged, the rest are judged."""

from types import SimpleNamespace

from evaluator.judge import core as jc


class _Cfg:
    max_cases = 0
    model = "m"
    judge_mode = "both"
    judge_aspects = ["relevance"]
    reference_mode = "free"
    include_doc_text = True
    judge_top_k = 5
    pass_threshold = 0.5
    score_aggregation = "average"
    aspect_weights = None
    system_prompt = None
    user_prompt_template = None

    def to_llm_config(self):
        return SimpleNamespace(
            timeout_s=1, model="m", api_base=None, temperature=0.0,
            api_key_env=None, use_local_server=False, local_server_url=None,
        )

    def get_api_base(self):
        return "http://x"


def test_one_failed_trace_does_not_abort_judging(monkeypatch):
    def fake_judge(trace, **kw):
        if trace.get("query_id") == "q2":
            raise TimeoutError("llm timed out")
        return {"overall": 1.0, "aspect_scores": {"relevance": 1.0},
                "verdict": "PASS", "reason": "ok"}

    monkeypatch.setattr(jc, "judge_trace", fake_judge)
    traces = [{"query_id": "q1"}, {"query_id": "q2"}, {"query_id": "q3"}]
    out = jc.run_llm_judging(traces, _Cfg())
    assert out["cases"] == 2  # q2 dropped, q1+q3 judged
    assert [d["query_id"] for d in out["details"]] == ["q1", "q3"]
    assert out["mean_score"] == 1.0
    assert out["pass_rate"] == 1.0
    assert out["aspect_means"]["relevance"] == 1.0
