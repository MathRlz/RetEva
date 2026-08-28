"""Retrieval depth must cover every published report cutoff.

With depth 5 and cutoffs @1/@5/@10, ``recall@10`` was ``retrieved[:10]`` over a 5-long list —
i.e. ``recall@5`` under a deeper label (seen in the committed pubmed200 campaign results, where
every ``@10`` equalled its ``@5``). The floor makes the deepest cutoff real.
"""

import re
from types import SimpleNamespace

import numpy as np

from tests.graph_test_helpers import make_state
from evaluator.config.vector_db import VectorDBConfig
from evaluator.evaluation.executor.run import _result_depth
from evaluator.evaluation.metric_registry import list_metrics
from evaluator.metrics.ir import IR_CUTOFFS, RETRIEVAL_DEPTH, report_depth


def test_depth_is_deepest_cutoff():
    assert RETRIEVAL_DEPTH == max(IR_CUTOFFS)


def test_no_registered_cutoff_exceeds_depth():
    # Registering an @20 metric without raising the depth would recreate the bug.
    cutoffs = [int(m.group(1)) for m in
               (re.search(r"@(\d+)$", spec.name) for spec in list_metrics()) if m]
    assert cutoffs, "no @k metrics registered"
    assert max(cutoffs) <= RETRIEVAL_DEPTH


def test_shallow_config_is_floored():
    assert report_depth(5) == RETRIEVAL_DEPTH
    assert report_depth(1) == RETRIEVAL_DEPTH
    assert _result_depth(5, "vector_db.k") == RETRIEVAL_DEPTH


def test_deeper_config_is_kept():
    assert report_depth(30) == 30
    assert _result_depth(30, "vector_db.k") == 30


def test_default_config_depth():
    assert VectorDBConfig().k == RETRIEVAL_DEPTH


def test_retrieval_node_asks_the_store_for_the_floored_depth():
    """The wiring check: a `k: 5` retrieval node must still fetch 10 candidates, else the
    floor is a constant nobody reads."""
    from evaluator.evaluation.handlers.retrieval import _stage_retrieval

    asked = {}

    class _Pipeline:
        needs_refinement = False
        strategy_config = SimpleNamespace(core=SimpleNamespace(mode="dense"))

        def retrieve_candidates(self, vectors, k, query_texts=None, mode=None):
            asked["k"] = k
            return [[({"doc_id": "d1"}, 1.0)] for _ in range(len(vectors))]

    s = make_state(k=5, retrieval_pipeline=_Pipeline())
    s.refine_in_graph = True  # candidate-fetch path: no finalize/rerank machinery needed
    s.current_node = SimpleNamespace(id="retrieval", params={"k": 5}, kind="retrieval")
    s.input = lambda key, default=None: (
        np.zeros((2, 3), dtype="float32") if key == "query_vectors" else default
    )
    s.get_artifact = lambda name, default=None: default
    s.input_items = lambda *_a, **_kw: None
    s.keyed_items = lambda *_a, **_kw: None

    _stage_retrieval(s)
    assert asked["k"] == RETRIEVAL_DEPTH
