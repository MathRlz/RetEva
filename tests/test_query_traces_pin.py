"""R3 pin: the trace builder's entry schema and id-join semantics.

`_build_query_traces` had ZERO direct coverage while the export layer pins its keys
(`per_query_wer`, `recall_at_5`). Written in Phase 0 against the positional side channels,
updated in Phase 2: the per-item scores and answer details now ride the keyed bus and are
joined by query id — the entry schema is unchanged, and a trace whose id has no score
simply omits that key instead of silently borrowing its positional neighbour's.
"""

from evaluator.evaluation.executor.state import RunState
from evaluator.evaluation.handlers.rag import _stage_build_query_traces
from evaluator.evaluation.item_set import ItemSet
from evaluator.pipeline.graph.registry import StageNode


class _DatasetStub:
    questions: list = []


def _state(trace_limit=2) -> RunState:
    from tests.graph_test_helpers import make_state

    s = make_state(
        dataset=_DatasetStub(), mode="asr_text_retrieval",
        retrieval_pipeline=object(), asr_pipeline=object(),
        trace_limit=trace_limit, total=3,
    )
    s.current_node = StageNode(
        id="build_query_traces",
        stage="measure",
        bindings=(("retrieved", "retrieval"),
                  ("reference_transcription", "dataset_source"),
                  ("relevant_docs", "dataset_source"),
                  ("per_query_wer", "transcription_metrics"),
                  ("per_query_cer", "transcription_metrics"),
                  ("per_query_recall5", "retrieval_metrics"),
                  ("generated_answers", "answer_gen")),
    )
    ids = ["q1", "q2", "q3"]
    s.ctx.put("retrieval", "retrieved", ItemSet(ids, [
        [({"doc_id": "d1", "text": "doc one"}, 0.9)],
        [({"doc_id": "d2", "text": "doc two"}, 0.8)],
        [({"doc_id": "d3", "text": "doc three"}, 0.7)],
    ]))
    s.ctx.put("dataset_source", "reference_transcription",
              ItemSet(ids, ["ref one", "ref two", "ref three"]))
    s.ctx.put("dataset_source", "relevant_docs",
              ItemSet(ids, [{"d1": 1}, {"d9": 1}, {"d3": 1}]))
    # R3-P2: per-item scores + answer details ride the keyed bus, joined by query id.
    s.ctx.put("transcription_metrics", "per_query_wer", ItemSet(ids, [0.1, 0.2, 0.3]))
    s.ctx.put("transcription_metrics", "per_query_cer", ItemSet(ids, [0.01, 0.02, 0.03]))
    s.ctx.put("retrieval_metrics", "per_query_recall5", ItemSet(ids, [1.0, 0.0, 1.0]))
    s.ctx.put("answer_gen", "generated_answers", ItemSet(
        ["q1"], [{"generated_answer": "a1", "reference_answer": "r1"}]
    ))
    return s


def test_trace_entries_pin_schema_and_pairing():
    s = _state(trace_limit=2)
    _stage_build_query_traces(s)
    traces = s.results["query_traces"]
    assert len(traces) == 2  # trace_limit caps

    first = traces[0]
    assert set(first) == {
        "query_id", "relevant", "retrieved", "question", "generated_answer",
        "reference_answer", "metadata", "per_query_wer", "per_query_cer", "recall_at_5",
    }
    assert first["query_id"] == "q1"
    assert first["relevant"] == {"d1": 1}
    assert first["retrieved"] == [{"doc_key": "d1", "score": 0.9, "text": "doc one"}]
    assert first["question"] == "ref one"
    assert first["generated_answer"] == "a1"
    assert first["per_query_wer"] == 0.1
    assert first["per_query_cer"] == 0.01
    assert first["recall_at_5"] == 1.0

    second = traces[1]
    assert second["query_id"] == "q2"
    assert second["generated_answer"] == ""  # no answer detail for q2
    assert second["per_query_wer"] == 0.2

    # The builder publishes the traces on the bus too (survives the metrics node's
    # `s.results` rebuild).
    assert s.ctx.get("build_query_traces", "query_traces") == traces
    # No dedup-by-shared-key guard (R4/multi-variant: a graph with N build_query_traces nodes,
    # one per compared variant, must build ALL N, not just the first) — a second call on the
    # SAME node id genuinely rebuilds from the bus's current state.
    s.ctx.put("transcription_metrics", "per_query_wer",
              ItemSet(["q1", "q2", "q3"], [9.9, 9.9, 9.9]))
    _stage_build_query_traces(s)
    assert s.results["query_traces"][0]["per_query_wer"] == 9.9


def test_trace_limit_zero_means_no_limit():
    """0 = NO LIMIT, matching what the same knob does to the dataset (`questions[:trace_limit]`
    only when positive) and what DataConfig documents. It used to mean "off", so one number had
    opposite meanings at the two ends of a run. Traces are disabled by omitting the node."""
    s = _state(trace_limit=0)
    _stage_build_query_traces(s)
    assert len(s.results["query_traces"]) == 3      # every query, not none


def test_trace_limit_caps_when_positive():
    s = _state(trace_limit=2)
    _stage_build_query_traces(s)
    assert len(s.results["query_traces"]) == 2


def test_scores_join_by_id_not_position():
    """R3-P2: a sparse score set lines up by id — the item without a score omits the key
    instead of borrowing its positional neighbour's value (the old zip-by-index bug)."""
    s = _state(trace_limit=3)
    # only q3 has a WER score, and it sits at position 0 of its own ItemSet
    s.ctx.put("transcription_metrics", "per_query_wer", ItemSet(["q3"], [0.7]))
    s.ctx.put("transcription_metrics", "per_query_cer", ItemSet(["q3"], [0.07]))
    _stage_build_query_traces(s)
    traces = {t["query_id"]: t for t in s.results["query_traces"]}
    assert "per_query_wer" not in traces["q1"]  # positionally it would have taken 0.7
    assert "per_query_wer" not in traces["q2"]
    assert traces["q3"]["per_query_wer"] == 0.7
    assert traces["q3"]["per_query_cer"] == 0.07
    assert traces["q1"]["recall_at_5"] == 1.0   # a full set still joins for everyone
