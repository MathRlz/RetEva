"""Roadmap 2d: item replay by query id — dataset slice + CLI trace."""

import json

from evaluator.cli.replay import _print_trace, parse_replay_args
from evaluator.datasets.runtime import slice_by_query_ids


class _Q:
    def __init__(self, qid):
        self.question_id = qid


class _FakeDataset:
    """A questions-backed QueryDataset stand-in (light id source = .questions)."""

    def __init__(self):
        self.questions = [_Q("q1"), _Q("q2"), _Q("q3")]
        self.corpus_entries = [{"doc_id": "d1", "text": "alpha"}]

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, i):
        return {"question_id": self.questions[i].question_id, "row": i}

    def get_corpus(self):
        return [{"doc_id": "d1", "text": "alpha"}, {"doc_id": "d2", "text": "beta"}]


def test_slice_keeps_corpus_whole_and_filters_queries():
    ds = _FakeDataset()
    sliced = slice_by_query_ids(ds, {"q2"})
    assert len(sliced) == 1
    assert sliced[0]["question_id"] == "q2"
    # corpus passes through unchanged (retrieval scores match a full run)
    assert len(sliced.get_corpus()) == 2
    # the id-bearing list is sliced too
    assert [q.question_id for q in sliced.questions] == ["q2"]


def test_slice_unknown_id_is_empty():
    assert len(slice_by_query_ids(_FakeDataset(), {"nope"})) == 0


def test_slice_multiple_ids_preserves_order():
    sliced = slice_by_query_ids(_FakeDataset(), {"q1", "q3"})
    assert [sliced[i]["question_id"] for i in range(len(sliced))] == ["q1", "q3"]


def test_subset_delegates_unknown_attrs_to_base():
    sliced = slice_by_query_ids(_FakeDataset(), {"q1"})
    # corpus_entries is not overridden → delegated to the base dataset
    assert sliced.corpus_entries == [{"doc_id": "d1", "text": "alpha"}]


def test_slice_falls_back_to_row_id_without_light_source():
    class _NoLightDS:
        def __len__(self):
            return 2

        def __getitem__(self, i):
            return {"question_id": f"x{i}"}

        def get_corpus(self):
            return []

    sliced = slice_by_query_ids(_NoLightDS(), {"x1"})
    assert len(sliced) == 1 and sliced[0]["question_id"] == "x1"


def test_print_trace_groups_by_node_order(tmp_path, capsys):
    # Two nodes' artifacts dumped as <node>.<artifact>.jsonl (one row each = single item).
    (tmp_path / "asr.query_text.jsonl").write_text(
        json.dumps({"id": "q1", "value": "hello world"}) + "\n"
    )
    (tmp_path / "retrieval.retrieved.jsonl").write_text(
        json.dumps({"id": "q1", "value": ["d2", "d1"]}) + "\n"
    )
    _print_trace(tmp_path, ["asr", "retrieval"], "q1")
    out = capsys.readouterr().out
    assert "Replay trace for query 'q1'" in out
    # node order honored: asr before retrieval
    assert out.index("[asr]") < out.index("[retrieval]")
    assert "query_text: 'hello world'" in out
    assert "retrieved: ['d2', 'd1']" in out


def test_print_trace_handles_no_dumps(tmp_path, capsys):
    _print_trace(tmp_path, ["asr"], "q9")
    assert "no artifacts dumped" in capsys.readouterr().out


def test_replay_arg_parsing():
    a = parse_replay_args(["--config", "c.yaml", "--query-id", "q42", "--dump-dir", "/tmp/d"])
    assert a.config == "c.yaml" and a.query_id == "q42" and a.dump_dir == "/tmp/d"
