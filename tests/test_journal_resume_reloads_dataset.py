"""A resumed run must still load the dataset.

The journal snapshots artifacts + results, NOT ``state.dataset``
(``run_journal._SNAPSHOT_FIELDS``). Skipping the source level on resume therefore left every
dataset reader (audio-embedding / asr stages, diagnostics) holding ``None``, surfacing much later
as `object of type 'NoneType' has no len()` inside the DataLoader — the crash reported for
`apm_admed_whisper_*`, and a "no query vectors" error in a RAG graph.
"""

import pytest

from evaluator.evaluation.executor.engine import _execute_stage_graph
from evaluator.evaluation.run_journal import RunJournal, run_key, snapshot_state
from evaluator.pipeline import build_graph_from_spec
from evaluator.storage.cache import CacheManager

from tests.graph_test_helpers import explicit_graph, make_state

_NODES = ["dataset_source", "corpus_embedding", "vector_db"]


def _graph():
    spec = explicit_graph(_NODES)
    return build_graph_from_spec(spec["nodes"], edges=spec["edges"])


def _state(tmp_path, ran):
    """State whose stage handlers only record which nodes ran (no models, no data)."""
    from evaluator.evaluation import stage_registry

    state = make_state(
        cache_manager=CacheManager(cache_dir=str(tmp_path / "cache"), enabled=False),
        experiment_id="journal_resume_test",
        checkpoint_interval=1,
        resume_from_checkpoint=True,
    )
    for stage in {n.stage for n in _graph().nodes}:
        spec = stage_registry.get_stage_spec(stage)
        object.__setattr__(spec, "fn", lambda s, _st=stage: ran.append(_st))
    return state


@pytest.fixture
def _restore_registry():
    from evaluator.evaluation import stage_registry

    saved = {s: stage_registry.get_stage_spec(s).fn
             for s in ("source", "embed", "index")}
    yield
    for stage, fn in saved.items():
        object.__setattr__(stage_registry.get_stage_spec(stage), "fn", fn)


def test_source_nodes_rerun_when_resuming(tmp_path, _restore_registry):
    graph = _graph()
    ran: list = []
    state = _state(tmp_path, ran)

    # A journal that says level 0 (the source level) already completed.
    key = run_key(state.config, tuple(n.id for n in graph.nodes))
    RunJournal(state.cache_manager.checkpoints_dir, key).save(0, snapshot_state(state))

    _execute_stage_graph(state, graph, None)

    # the source stage re-runs (it owns the dataset load) while the rest of level 0 is skipped
    assert "source" in ran, f"source node was skipped on resume; ran={ran}"


def test_dataset_is_not_snapshotted():
    # If the dataset ever becomes part of the snapshot, the re-run above is redundant — but
    # until then, skipping the source level silently drops it.
    from evaluator.evaluation.run_journal import _SNAPSHOT_FIELDS

    assert "dataset" not in _SNAPSHOT_FIELDS
