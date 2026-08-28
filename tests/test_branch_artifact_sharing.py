"""P4 pin: a lone BRANCH-NAMESPACED producer is not shared across branches.

Found by an Ollama smoke run: with exactly one corrector branch (`asr` baseline vs `llm`),
both branches reported identical corrected_wer/cer/ceer_rx — the single
`query_correction@llm` publisher was treated as a global producer, so the uncorrected
baseline was scored against the corrector's text and the comparison read as zero effect.
A genuinely global producer (un-namespaced id) must still be shared.
"""

from evaluator.evaluation.handlers.metrics import _branched_items
from evaluator.evaluation.item_set import ItemSet
from tests.graph_test_helpers import make_state


def _state_with(producer_ids, artifact):
    s = make_state()

    class _Node:
        id = "aggregate"
        bindings = tuple((artifact, pid) for pid in producer_ids)

    s.current_node = _Node()
    for pid in producer_ids:
        s.ctx.put(pid, artifact, ItemSet(["q1"], [f"text from {pid}"]))
    return s


def test_lone_branch_producer_of_exclusive_artifact_is_not_shared():
    s = _state_with(["query_correction@llm"], "corrected_query_text")
    by_branch, only_shared = _branched_items(
        s, "corrected_query_text", branch_exclusive=True
    )
    assert set(by_branch) == {"llm"}
    assert only_shared is None  # the `asr` baseline must NOT inherit the corrector's text


def test_lone_global_producer_is_still_shared():
    s = _state_with(["query_correction"], "corrected_query_text")
    by_branch, only_shared = _branched_items(
        s, "corrected_query_text", branch_exclusive=True
    )
    assert set(by_branch) == {"main"}
    assert only_shared is not None  # one un-namespaced producer genuinely serves everyone


def test_cse_collapsed_producer_still_serves_every_branch():
    # `asr@asr` is ONE node CSE shares across branches — query_text is not branch-exclusive,
    # so every branch must keep reading it (this is what the c7 parity gate exercises).
    s = _state_with(["asr@asr"], "query_text")
    by_branch, only_shared = _branched_items(s, "query_text")
    assert only_shared is not None


def test_multiple_branch_producers_stay_per_branch():
    s = _state_with(["query_correction@rule", "query_correction@kb"], "corrected_query_text")
    by_branch, only_shared = _branched_items(
        s, "corrected_query_text", branch_exclusive=True
    )
    assert set(by_branch) == {"rule", "kb"}
    assert only_shared is None
