"""Branch fail-fast: a node whose handler raises marks its branch failed and skips every
transitive dependent, while sibling branches keep running to completion.

Before this, `_run_one_node` had try/finally with no except — one broken node (e.g. a TTS
engine with a missing voice) aborted the whole run and discarded every healthy variant's
results. And the TTS worker (`synthesize_missing_query_audio`) never raised at all: per-clip
failures were logged and swallowed, the node "succeeded" publishing nothing, and the run died
much later inside ASR with a misleading `Question ... has no audio_path`.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from evaluator.evaluation.executor.engine import _execute_stage_graph
from evaluator.evaluation import stage_registry
from evaluator.pipeline.graph.registry import StageGraph, StageNode

from tests.graph_test_helpers import make_state


def _two_branch_graph():
    """src -> (a1 -> b1 -> c1) and (a2 -> b2 -> c2): two independent 3-node branches."""
    nodes = (
        StageNode(id="src", stage="source"),
        StageNode(id="a1", stage="embed", depends_on=("src",)),
        StageNode(id="a2", stage="embed", depends_on=("src",)),
        StageNode(id="b1", stage="search", depends_on=("a1",)),
        StageNode(id="b2", stage="search", depends_on=("a2",)),
        StageNode(id="c1", stage="measure", depends_on=("b1",)),
        StageNode(id="c2", stage="measure", depends_on=("b2",)),
    )
    return StageGraph(mode="custom", nodes=nodes)


_STAGES = ("source", "embed", "search", "measure")


@pytest.fixture
def _fake_handlers():
    """Replace the real stage handlers with id-recording fakes; restore afterwards."""
    saved = {s: stage_registry.get_stage_spec(s).fn for s in _STAGES}
    ran: list = []
    failing: set = set()

    def _fn(s):
        nid = s.current_node.id
        ran.append(nid)
        if nid in failing:
            raise RuntimeError(f"boom in {nid}")

    for s in _STAGES:
        object.__setattr__(stage_registry.get_stage_spec(s), "fn", _fn)
    yield ran, failing
    for s, fn in saved.items():
        object.__setattr__(stage_registry.get_stage_spec(s), "fn", fn)


def test_failed_branch_stops_at_error_while_sibling_completes(_fake_handlers):
    ran, failing = _fake_handlers
    failing.add("a1")
    state = make_state()

    _execute_stage_graph(state, _two_branch_graph(), None)

    assert "a1" in ran
    # nothing downstream of the failure ran…
    assert "b1" not in ran and "c1" not in ran
    # …but the sibling branch ran to the end
    assert {"src", "a2", "b2", "c2"}.issubset(set(ran))
    assert list(state.failed_nodes) == ["a1"]
    assert "RuntimeError" in state.failed_nodes["a1"]
    # transitive skip, each attributed to the ROOT failed node (not the nearest skipped one)
    assert state.skipped_nodes == {"b1": "a1", "c1": "a1"}


def test_mid_branch_failure_skips_only_its_own_tail(_fake_handlers):
    ran, failing = _fake_handlers
    failing.add("b2")
    state = make_state()

    _execute_stage_graph(state, _two_branch_graph(), None)

    assert "c2" not in ran
    assert {"src", "a1", "b1", "c1", "a2", "b2"}.issubset(set(ran))
    assert list(state.failed_nodes) == ["b2"]
    assert state.skipped_nodes == {"c2": "b2"}


def test_keyboard_interrupt_still_aborts_the_run(_fake_handlers):
    ran, _ = _fake_handlers

    def _interrupt(s):
        raise KeyboardInterrupt()

    object.__setattr__(stage_registry.get_stage_spec("embed"), "fn", _interrupt)
    state = make_state()

    with pytest.raises(KeyboardInterrupt):
        _execute_stage_graph(state, _two_branch_graph(), None)
    assert not state.failed_nodes  # a user abort is not a "failed branch"


def test_progress_sink_sees_error_and_skip_events(_fake_handlers, monkeypatch, tmp_path):
    ran, failing = _fake_handlers
    failing.add("a1")
    progress = tmp_path / "progress.jsonl"
    monkeypatch.setenv("EVALUATOR_PROGRESS_FILE", str(progress))
    state = make_state()

    _execute_stage_graph(state, _two_branch_graph(), None)

    import json

    events = [json.loads(line) for line in progress.read_text().splitlines()]
    by_node = {}
    for e in events:
        by_node.setdefault(e["node"], []).append(e["event"])
    assert "node_error" in by_node["a1"] and "node_complete" not in by_node["a1"]
    assert by_node["b1"] == ["node_skipped"]
    assert "node_complete" in by_node["c2"]


def test_tts_synthesis_failure_raises_instead_of_publishing_holes(tmp_path, monkeypatch):
    from evaluator.config.audio_synthesis import AudioSynthesisConfig
    from evaluator.pipeline.audio import prepare as prepare_mod
    from evaluator.storage.cache.manager import CacheManager

    class _HalfBrokenSynthesizer:
        def __init__(self, config):
            self.config = config

        def synthesize(self, text, output_path=None):
            if "diagnosis" in text:
                raise RuntimeError("no voice model")
            if output_path:
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                Path(output_path).write_bytes(b"fake-wav")

    monkeypatch.setattr(
        "evaluator.pipeline.audio.synthesis.AudioSynthesizer", _HalfBrokenSynthesizer,
    )
    questions = [
        SimpleNamespace(question_id="q1", question_text="what is the treatment", audio_path=None),
        SimpleNamespace(question_id="q2", question_text="what is the diagnosis", audio_path=None),
    ]
    cache = CacheManager(cache_dir=str(tmp_path), enabled=True)
    cfg = AudioSynthesisConfig(provider="piper", language="en", output_dir=str(tmp_path / "audio"))

    with pytest.raises(RuntimeError, match=r"failed for 1/2 question\(s\)"):
        prepare_mod.synthesize_missing_query_audio(questions, cfg, cache_manager=cache)
