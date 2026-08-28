"""A4 pins (CRITIQUE.md): asymmetric per-branch drops mark the paired delta drop-biased.

The paired delta is computed over the id intersection; items usually leave it because a
branch dropped them, and hard items fail more (not-missing-at-random). Beyond
``DROP_BIAS_THRESHOLD`` one-sided exclusions the delta carries ``drop_biased: true`` (key
absent otherwise — clean reports are byte-unchanged).
"""

from evaluator.evaluation.aggregate import build_report, paired_delta
from evaluator.evaluation.item_set import ItemSet


def _items(ids, value=0.5):
    return ItemSet(ids, [value] * len(ids))


def test_full_overlap_is_not_flagged():
    delta = paired_delta(_items(["q1", "q2", "q3"]), _items(["q1", "q2", "q3"]))
    assert "drop_biased" not in delta


def test_asymmetric_exclusions_flag_the_delta():
    # 9 paired, 1 only-branch + 1 only-baseline = 2 excluded > 5% of 9.
    branch = _items([f"q{i}" for i in range(9)] + ["q_only_branch"])
    baseline = _items([f"q{i}" for i in range(9)] + ["q_only_base"])
    delta = paired_delta(branch, baseline)
    assert delta["n_paired"] == 9
    assert delta["drop_biased"] is True


def test_empty_intersection_is_flagged():
    delta = paired_delta(_items(["a"]), _items(["b"]))
    assert delta["n_paired"] == 0
    assert delta["drop_biased"] is True


def test_flag_survives_into_the_report_deltas():
    per_branch = {
        "base": {"wer": _items([f"q{i}" for i in range(9)] + ["only_base"], 0.2)},
        "corr": {"wer": _items([f"q{i}" for i in range(9)] + ["only_corr"], 0.1)},
    }
    report = build_report(per_branch, baseline="base")
    assert report["deltas"]["corr_vs_base"]["wer"]["drop_biased"] is True
