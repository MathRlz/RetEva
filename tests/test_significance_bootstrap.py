"""C2/F21: bootstrap CI uses a local RNG (no global-state perturbation) and the A/B CIs in
compare_experiments draw from independent seeds (42 vs 43), not the same one.
"""

import numpy as np

from evaluator.analysis.significance import (
    bootstrap_confidence_interval,
    compare_experiments,
)


def test_bootstrap_is_reproducible_per_seed():
    scores = [0.1, 0.4, 0.2, 0.9, 0.3]
    assert bootstrap_confidence_interval(scores, random_state=7) == \
        bootstrap_confidence_interval(scores, random_state=7)


def test_bootstrap_does_not_perturb_global_rng():
    # capture the global stream, run a seeded CI in between, expect the stream unchanged
    np.random.seed(123)
    before = np.random.rand(4)
    np.random.seed(123)
    _ = bootstrap_confidence_interval([0.0, 1.0, 0.0, 1.0], random_state=99)
    after = np.random.rand(4)
    assert np.allclose(before, after)


# a varied continuous-ish sample so different resamples actually move the percentile bounds
_VARIED = [0.05, 0.21, 0.33, 0.48, 0.62, 0.71, 0.88, 0.95, 0.12, 0.40, 0.57, 0.83]


def test_different_seeds_give_different_resamples():
    assert bootstrap_confidence_interval(_VARIED, random_state=42) != \
        bootstrap_confidence_interval(_VARIED, random_state=43)


def test_compare_experiments_ci_a_and_b_independent():
    # identical per-sample scores for A and B; with the OLD shared seed ci_a == ci_b
    # exactly. Independent seeds (42 vs 43) must break that tie.
    res_a = {"per_sample": {"MRR": _VARIED}}
    res_b = {"per_sample": {"MRR": list(_VARIED)}}
    out = compare_experiments(res_a, res_b, metric_names=["MRR"])["MRR"]
    assert "ci_a" in out and "ci_b" in out
    assert out["ci_a"] != out["ci_b"]


def test_compare_experiments_applies_fdr_across_metric_panel():
    # M2: comparing many metrics must report BH-FDR q-values + an FDR-corrected flag, not just
    # raw per-metric p < 0.05 (which inflates the family-wise false-positive rate).
    import numpy as np

    from evaluator.analysis.significance import compare_experiments

    rng = np.random.RandomState(0)
    n = 40
    metrics = ["m1", "m2", "m3", "m4", "m5"]
    res_a = {"per_sample": {}}
    res_b = {"per_sample": {}}
    for k in metrics:
        a = list(rng.rand(n))
        b = [x + 0.3 for x in a] if k == "m1" else list(rng.rand(n))  # m1 truly shifted
        res_a[k] = float(np.mean(a))
        res_a["per_sample"][k] = a
        res_b[k] = float(np.mean(b))
        res_b["per_sample"][k] = b

    out = compare_experiments(res_a, res_b, metric_names=metrics)
    # q-values present + monotone w.r.t. raw p (q >= p for every tested metric)
    for r in out.values():
        tt = r["ttest"]
        assert "q_value" in tt
        assert tt["q_value"] >= tt["p_value"] - 1e-9
        assert "significant_ttest_fdr" in r
    # the genuine effect survives FDR; pure-noise metrics are not FDR-significant
    assert out["m1"]["significant_ttest_fdr"] is True
    assert all(not out[k]["significant_ttest_fdr"] for k in ["m2", "m3", "m4", "m5"])


def test_underpowered_flag_and_report_rendering():
    # R3: small paired sample → underpowered flagged; the text report surfaces p, q (FDR),
    # FDR-significance, and the under-powered warning.
    from evaluator.analysis.significance import format_comparison_report

    a = {"MRR": 0.3, "per_sample": {"MRR": [0.0, 0.25, 0.5, 0.75, 1.0]}}
    b = {"MRR": 0.5, "per_sample": {"MRR": [0.2, 0.45, 0.7, 0.95, 1.2]}}
    cmp = compare_experiments(a, b, metric_names=["MRR"])
    assert cmp["MRR"]["n_samples"] == 5
    assert cmp["MRR"]["underpowered"] is True
    text = format_comparison_report({"metrics": cmp})
    assert "q=" in text  # FDR q-value surfaced
    assert "under-powered" in text


def test_not_underpowered_above_threshold():
    import numpy as np
    a = {"per_sample": {"MRR": list(np.linspace(0, 1, 25))}}
    b = {"per_sample": {"MRR": list(np.linspace(0.1, 1.1, 25))}}
    assert compare_experiments(a, b, metric_names=["MRR"])["MRR"]["underpowered"] is False
