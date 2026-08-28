"""Dataset loading is an in-graph action now: the `dataset_source` node's `_ensure_dataset_loaded`
owns the load (single + multi source + replay slice), idempotently, stashing the loaded dataset on
the run's `load_info` carrier. The full end-to-end load is covered by the integration tests + the
container parity gate; this pins the loader mechanics with a duck-typed state."""
from types import SimpleNamespace

from evaluator.evaluation.handlers import source as src
from evaluator.evaluation.load_info import LoadInfo


def _state(**over):
    base = dict(
        dataset=None, config=object(), load_info=LoadInfo(), dataset_sources={},
        disable_ir_metrics=False, join_warning="", total=0,
    )
    base.update(over)
    return SimpleNamespace(**base)


def test_loads_in_node_and_sets_total_and_carrier(monkeypatch):
    monkeypatch.setattr(
        "evaluator.datasets.runtime.load_runtime_dataset", lambda cfg: ["a", "b", "c"]
    )
    monkeypatch.setattr(
        "evaluator.datasets.runtime.load_dataset_sources", lambda cfg: ({}, False, "")
    )
    s = _state()
    src._ensure_dataset_loaded(s)
    assert list(s.dataset) == ["a", "b", "c"]
    assert s.total == 3
    assert s.load_info.dataset == ["a", "b", "c"]  # carrier out (for num_samples)


def test_idempotent_no_reload(monkeypatch):
    # dataset already set (a later source node, or a back-compat pre-loaded run) → no-op.
    def _boom(cfg):
        raise AssertionError("reloaded an already-loaded dataset")

    monkeypatch.setattr("evaluator.datasets.runtime.load_runtime_dataset", _boom)
    s = _state(dataset=["x"])
    src._ensure_dataset_loaded(s)  # must not raise
    assert list(s.dataset) == ["x"]


def test_replay_slice_applied_at_load(monkeypatch):
    monkeypatch.setattr(
        "evaluator.datasets.runtime.load_runtime_dataset", lambda cfg: ["q0", "q1", "q2"]
    )
    monkeypatch.setattr(
        "evaluator.datasets.runtime.load_dataset_sources", lambda cfg: ({}, False, "")
    )
    monkeypatch.setattr(
        "evaluator.datasets.runtime.slice_by_query_ids", lambda ds, qids: ["q1"]
    )
    s = _state(load_info=LoadInfo(replay_query_ids=["q1"]))
    src._ensure_dataset_loaded(s)
    assert list(s.dataset) == ["q1"] and s.total == 1


def test_multi_source_join_gate_carried(monkeypatch):
    monkeypatch.setattr(
        "evaluator.datasets.runtime.load_runtime_dataset", lambda cfg: ["a", "b"]
    )
    monkeypatch.setattr(
        "evaluator.datasets.runtime.load_dataset_sources",
        lambda cfg: ({"qa": ["a"], "docs": ["d"]}, True, "disjoint join"),
    )
    s = _state()
    src._ensure_dataset_loaded(s)
    assert s.dataset_sources == {"qa": ["a"], "docs": ["d"]}
    assert s.disable_ir_metrics is True and s.join_warning == "disjoint join"


def test_builder_preview_on_dataset_source_rooted_graph():
    # The builder/preview is descriptor-driven (no data load), so a dataset_source-rooted graph
    # still renders. resolve_node_form for dataset_source exposes the dataset picker.
    from evaluator.webapi.form_builder import resolve_node_form

    form = resolve_node_form("dataset_source", {})
    assert form["type"] in ("dataset_source", "source")
    assert form["inputs"] == []  # the source is the root: no inputs
