"""A1 pins (CRITIQUE.md): lineage fan-out is statistically sound.

Variants (``q42·aug0/1``) roll up to one parent-level score before pairing/reduction —
the cluster-safe unit — and a variant's GT resolves via its lineage parent, so augmented
items are scored (not silently dropped) in the branched aggregate path.
"""

from evaluator.evaluation.aggregate import build_report, rollup_variants
from evaluator.evaluation.item_set import ItemSet, lineage_parent
from evaluator.evaluation.metric_registry import compute_metric, get_metric


def test_lineage_parent_identity_for_plain_ids():
    assert lineage_parent("q42·aug0") == "q42"
    assert lineage_parent("q42") == "q42"


def test_rollup_is_identity_without_variants():
    scores = ItemSet(["q1", "q2"], [0.5, 1.0])
    assert rollup_variants(scores) is scores


def test_rollup_groups_by_parent_mean_min_max():
    scores = ItemSet(["q1·aug0", "q1·aug1", "q2·aug0"], [0.0, 1.0, 0.25])
    mean = rollup_variants(scores)
    assert mean.ids == ["q1", "q2"]
    assert mean.values == [0.5, 0.25]
    assert rollup_variants(scores, "min").values == [0.0, 0.25]
    assert rollup_variants(scores, "max").values == [1.0, 0.25]


def test_metric_gt_joins_variants_via_lineage_parent():
    artifacts = {
        "query_text": ItemSet(["q1·aug0", "q1·aug1"], ["hello world", "hello word"]),
        "reference_transcription": ItemSet(["q1"], ["hello world"]),
    }
    scores = compute_metric(get_metric("wer"), artifacts)
    assert scores.ids == ["q1·aug0", "q1·aug1"]  # variants stay distinct items
    assert scores.values[0] == 0.0
    assert scores.values[1] > 0.0


def test_variant_rollup_knob_flows_through_build_report():
    # A1 knob: build_report reduces variants with the configured reducer.
    augmented = {"wer": ItemSet(["q1·aug0", "q1·aug1"], [0.2, 0.6])}
    report = build_report({"aug": augmented}, variant_rollup="max")
    assert report["branches"]["aug"]["wer"]["mean"] == 0.6
    report = build_report({"aug": augmented}, variant_rollup="min")
    assert report["branches"]["aug"]["wer"]["mean"] == 0.2


def test_variant_rollup_config_validation_rejects_unknown_reducer():
    from evaluator.config.validation import _validate_variant_rollup

    errors = []
    _validate_variant_rollup(type("C", (), {"variant_rollup": "median"})(), errors)
    assert len(errors) == 1 and "median" in errors[0]
    errors = []
    _validate_variant_rollup(type("C", (), {"variant_rollup": "mean"})(), errors)
    assert errors == []


def test_paired_delta_pairs_augmented_branch_with_clean_baseline():
    clean = {"wer": ItemSet(["q1", "q2"], [0.0, 0.2])}
    augmented = {"wer": ItemSet(["q1·aug0", "q1·aug1", "q2·aug0"], [0.2, 0.4, 0.2])}
    report = build_report({"clean": clean, "aug": augmented}, baseline="clean")
    delta = report["deltas"]["aug_vs_clean"]["wer"]
    assert delta["n_paired"] == 2  # parent-level n, not 3 variant rows
    assert abs(delta["mean_delta"] - 0.15) < 1e-12  # ((0.3 − 0.0) + (0.2 − 0.2)) / 2
    assert report["branches"]["aug"]["wer"]["n"] == 2  # per-branch means are parent-level too
