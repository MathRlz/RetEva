"""Roadmap §7 — observability: failure attribution, typed ProgressEvent, artifact dump."""

import json

from evaluator.evaluation.artifact_dump import maybe_dump_node_artifacts
from evaluator.evaluation.item_isolation import DropSink, isolate_batch
from evaluator.evaluation.item_set import ItemSet
from evaluator.evaluation.progress import ProgressEvent, ProgressSink


def test_failure_attribution_summary():
    sink = DropSink()

    def item_fn(v):
        if v in (2, 4):
            raise RuntimeError("clip too long")
        return v

    def batch_fn(values):
        raise RuntimeError("batch failed")

    out = isolate_batch(
        ["q1", "q2", "q3", "q4"], [1, 2, 3, 4], batch_fn, item_fn,
        node_id="audio_embedding", placeholder=None, sink=sink,
    )
    assert out == [1, None, 3, None]
    fs = sink.failure_summary()
    assert fs["total_dropped"] == 2
    assert fs["by_node"]["audio_embedding"] == 2
    assert fs["top_error_types"] == [("RuntimeError", 2)]
    assert fs["examples"][0]["error"] == "clip too long"


def test_clean_run_has_empty_failure_summary():
    assert DropSink().failure_summary() == {}


def test_progress_event_record_is_flat_and_matches_emit(tmp_path):
    ev = ProgressEvent(ts=1.0, event="node_complete", node="asr", level=2, total_levels=5,
                       extra={"duration_s": 0.3})
    assert ev.to_record()["duration_s"] == 0.3
    path = tmp_path / "p.jsonl"
    sink = ProgressSink(path=str(path), total_levels=3)
    sink.emit("node_complete", node="asr", level=1, duration_s=0.5)
    rec = json.loads(path.read_text().splitlines()[0])
    assert rec["node"] == "asr" and rec["duration_s"] == 0.5 and rec["total_levels"] == 3


def test_artifact_dump_is_env_gated(tmp_path, monkeypatch):
    class _Ctx:
        def __init__(self):
            self._s = {("embed", "query_vectors"): ItemSet(["q1", "q2"], [[1, 2], [3, 4]])}

        def slots(self):
            return list(self._s.keys())

        def get_opt(self, p, n, d=None):
            return self._s.get((p, n), d)

    from types import SimpleNamespace

    state = SimpleNamespace(ctx=_Ctx())
    node = SimpleNamespace(id="embed")

    # not listed → no dump
    monkeypatch.delenv("EVALUATOR_DUMP_ARTIFACTS", raising=False)
    maybe_dump_node_artifacts(state, node)
    assert not list(tmp_path.iterdir())

    # listed → dump per-item rows
    monkeypatch.setenv("EVALUATOR_DUMP_ARTIFACTS", "embed")
    monkeypatch.setenv("EVALUATOR_DUMP_DIR", str(tmp_path))
    maybe_dump_node_artifacts(state, node)
    out = tmp_path / "embed.query_vectors.jsonl"
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert [r["id"] for r in rows] == ["q1", "q2"]
    assert rows[0]["value"] == [1, 2]
