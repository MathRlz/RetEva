"""V3 pins: chart shaping is pure, JSON-safe, and never raises on a thin report.

The shaping is where the logic lives (the JS is a thin renderer), so this is where the
tests go.
"""

import json

import pytest

from evaluator.webapi.chart_data import (
    _num,
    delta_forest,
    denominator_bars,
    empty_spec,
)

_REPORT = {
    "deltas": {
        "asr_vs_ref": {
            "recall@5": {"mean_delta": -0.20, "ci": [-0.25, -0.15], "cohens_d": -0.53,
                         "p_value": 1e-10, "p_value_fdr": 2e-10, "n_paired": 200,
                         "n_only_branch": 0, "n_only_baseline": 0},
            "wer": {"mean_delta": 0.12, "ci": [0.10, 0.13], "cohens_d": 1.28,
                    "p_value_fdr": 5e-32, "n_paired": 200,
                    "n_only_branch": 30, "n_only_baseline": 0, "drop_biased": True},
        }
    }
}


def test_num_maps_non_finite_to_none():
    # JSON.parse rejects bare NaN/Infinity, so these must never reach the spec
    assert _num(float("nan")) is None
    assert _num(float("inf")) is None
    assert _num(float("-inf")) is None
    assert _num(0.5) == 0.5
    assert _num(None) is None
    assert _num("x") is None
    assert _num(True) is None  # a bool is not a measurement


def test_specs_are_json_parseable_even_with_degenerate_cis():
    """A zero-variance branch yields NaN bounds; the spec must still round-trip."""
    report = {"deltas": {"a_vs_b": {"mrr": {
        "mean_delta": 0.0, "ci": [float("nan"), float("nan")],
        "cohens_d": float("nan"), "p_value_fdr": float("nan"), "n_paired": 1,
    }}}}
    for spec in (delta_forest(report), denominator_bars(report)):
        text = json.dumps(spec)          # would raise ValueError on a non-finite float
        assert "NaN" not in text and "Infinity" not in text
        json.loads(text)                 # and JSON.parse's Python twin accepts it


def test_forest_orders_by_absolute_effect_size():
    spec = delta_forest(_REPORT)
    labels = spec["series"][0]["y"]
    # |d| = 1.28 (wer) outranks 0.53 (recall@5)
    assert labels[0].startswith("Δwer")
    assert labels[1].startswith("Δrecall@5")


def test_forest_whiskers_are_clamped_non_negative():
    """An inverted CI must not draw a negative whisker."""
    report = {"deltas": {"a_vs_b": {"m": {
        "mean_delta": 0.5, "ci": [0.9, 0.1],  # deliberately inverted
    }}}}
    err = delta_forest(report)["series"][0]["error_x"]
    assert min(err["array"]) >= 0 and min(err["arrayminus"]) >= 0


def test_forest_table_mirrors_the_plotted_values():
    spec = delta_forest(_REPORT)
    plotted = spec["series"][0]["x"]
    from_table = [row[1] for row in spec["table"]["rows"]]
    assert plotted == from_table       # the table twin cannot drift from the chart


def test_denominator_flags_drop_bias():
    spec = denominator_bars(_REPORT)
    flagged = [y for y in spec["series"][0]["y"] if "⚠" in y]
    assert len(flagged) == 1 and flagged[0].startswith("Δwer")
    assert "5%" in spec["note"]
    assert spec["layout"]["barmode"] == "stack"


def test_empty_reports_explain_themselves_instead_of_raising():
    for spec in (delta_forest({}), denominator_bars({})):
        assert spec["empty"] and isinstance(spec["empty"], str)
        assert spec["series"] == []
    assert empty_spec("i", "t", "why")["empty"] == "why"


def test_specs_carry_no_color_literals():
    """Colors are semantic keys resolved against the theme; a hex would freeze the
    chart in one mode."""
    for spec in (delta_forest(_REPORT), denominator_bars(_REPORT)):
        assert "#" not in json.dumps(spec)


def test_deltas_without_a_numeric_mean_are_skipped():
    report = {"deltas": {"a_vs_b": {"broken": {"ci": [0, 1]}, "ok": {"mean_delta": 0.1}}}}
    assert delta_forest(report)["series"][0]["y"] == ["Δok (a_vs_b)"]


def test_forest_keeps_deltas_that_lack_an_effect_size():
    report = {"deltas": {"a_vs_b": {
        "has_d": {"mean_delta": 0.1, "cohens_d": 0.2},
        "no_d": {"mean_delta": 0.9},
    }}}
    labels = delta_forest(report)["series"][0]["y"]
    assert len(labels) == 2 and labels[-1].startswith("Δno_d")  # sorted last, not dropped


def test_volcano_splits_significant_by_symbol_and_color():
    """Significance must never be color-alone (CVD / print)."""
    from evaluator.webapi.chart_data import delta_volcano

    report = {"deltas": {"a_vs_b": {
        "sig": {"mean_delta": 0.3, "p_value_fdr": 1e-6},
        "ns": {"mean_delta": 0.01, "p_value_fdr": 0.4},
    }}}
    spec = delta_volcano(report)
    by_name = {s["name"]: s for s in spec["series"]}
    assert set(by_name) == {"q < 0.05", "not significant"}
    assert by_name["q < 0.05"]["marker"]["symbol"] == "circle"
    assert by_name["not significant"]["marker"]["symbol"] == "x"
    assert by_name["q < 0.05"]["colorkey"] != by_name["not significant"]["colorkey"]


def test_volcano_clamps_zero_q_instead_of_plotting_infinity():
    """q=0 underflows to -log10(0)=inf, which JSON cannot carry and Plotly cannot draw."""
    from evaluator.webapi.chart_data import delta_volcano

    spec = delta_volcano({"deltas": {"a_vs_b": {"m": {"mean_delta": 1.0,
                                                      "p_value_fdr": 0.0}}}})
    y = spec["series"][0]["y"][0]
    assert y is not None and y > 0
    assert "Infinity" not in json.dumps(spec)


def test_volcano_needs_p_values():
    from evaluator.webapi.chart_data import delta_volcano

    spec = delta_volcano({"deltas": {"a_vs_b": {"m": {"mean_delta": 0.1}}}})
    assert "significance testing" in spec["empty"]


# ── V5: retrieval diagnostics ────────────────────────────────────────────────

_FAIL = {
    "rank_distribution": {"3-5": 12, "1": 40, "not_found": 8, "2": 20, "6+": 4},
    "failure_categories": {"corpus_gap": 5, "asr_failure": 9, "embedding_mismatch": 2},
}


def test_rank_buckets_keep_their_scale_order_not_dict_order():
    """Ranks are ordinal: bucket order carries meaning and must not follow insertion."""
    from evaluator.webapi.chart_data import RANK_BUCKETS, rank_distribution

    spec = rank_distribution(_FAIL)
    assert spec["series"][0]["x"] == list(RANK_BUCKETS)
    assert spec["series"][0]["y"] == [40, 20, 12, 4, 8]


def test_rank_uses_an_ordinal_ramp_with_failure_called_out():
    from evaluator.webapi.chart_data import rank_distribution

    keys = rank_distribution(_FAIL)["series"][0]["colorkey"]
    assert keys[:4] == ["ord1", "ord2", "ord3", "ord4"]   # light→dark = further down
    assert keys[-1] == "neg"                              # not_found is a failure, not a step


def test_rank_share_column_sums_to_one_hundred():
    from evaluator.webapi.chart_data import rank_distribution

    rows = rank_distribution(_FAIL)["table"]["rows"]
    total = sum(float(r[2].rstrip("%")) for r in rows)
    assert abs(total - 100.0) < 0.05


def test_failure_categories_sorted_and_zero_free():
    from evaluator.webapi.chart_data import failure_categories

    spec = failure_categories({"failure_categories": {"a": 0, "b": 3, "c": 7}})
    assert spec["series"][0]["y"] == ["b", "c"]      # ascending → largest on top
    assert 0 not in spec["series"][0]["x"]           # empty causes are not drawn


def test_recall_loss_orders_by_k_and_splits_pairs():
    from evaluator.webapi.chart_data import recall_loss_by_k

    report = {"retrieval_wer_impact": {
        "asr_vs_ref": {"recall@10": {"mean": 0.3}, "recall@1": {"mean": 0.1},
                       "recall@5": {"mean": 0.2}, "mrr": {"mean": 0.9}},
        "corr_vs_ref": {"recall@1": {"mean": 0.05}},
    }}
    spec = recall_loss_by_k(report)
    first = spec["series"][0]
    assert first["name"] == "asr_vs_ref"
    assert first["x"] == [1, 5, 10]                  # sorted by k, non-@k metrics dropped
    assert first["y"] == [0.1, 0.2, 0.3]
    assert len(spec["series"]) == 2


def test_diagnostics_explain_themselves_when_absent():
    from evaluator.webapi.chart_data import (failure_categories, rank_distribution,
                                             recall_loss_by_k)

    assert rank_distribution({})["empty"]
    assert failure_categories({})["empty"]
    assert recall_loss_by_k({})["empty"]


# ── V6: per-query distributions ──────────────────────────────────────────────

def _traces(n=50):
    return [{"query_id": f"q{i}",
             "per_query_wer": (i % 10) / 10.0,
             "recall_at_5": 1.0 if i % 3 else 0.0,
             "relevant": {"d1": 1},
             "retrieved": [{"doc_key": "d1", "score": 0.9 - i * 0.01},
                           {"doc_key": "d2", "score": 0.5}]}
            for i in range(n)]


def test_histogram_bins_server_side_and_keeps_every_query():
    from evaluator.webapi.chart_data import per_query_histogram

    spec = per_query_histogram(_traces(50), bins=10)
    assert len(spec["series"][0]["y"]) == 10
    assert sum(spec["series"][0]["y"]) == 50        # nothing lost, incl. the max value
    assert len(spec["table"]["rows"]) == 10


def test_histogram_survives_a_single_repeated_value():
    """A degenerate range must not divide by zero."""
    from evaluator.webapi.chart_data import per_query_histogram

    spec = per_query_histogram([{"per_query_wer": 0.2}] * 5, bins=10)
    assert sum(spec["series"][0]["y"]) == 5
    assert json.dumps(spec)


def test_histogram_needs_traces():
    from evaluator.webapi.chart_data import per_query_histogram

    assert "trace_limit" in per_query_histogram([])["empty"]


def test_scatter_caps_points_and_says_so():
    from evaluator.webapi import chart_data as C

    spec = C.wer_vs_recall_scatter(_traces(5000))
    assert len(spec["series"][0]["x"]) <= C._SCATTER_CAP
    assert "of 5000" in spec["note"]


def test_scatter_jitter_is_deterministic_and_keeps_true_values_for_hover():
    from evaluator.webapi.chart_data import wer_vs_recall_scatter

    a = wer_vs_recall_scatter(_traces(30))
    b = wer_vs_recall_scatter(_traces(30))
    assert a["series"][0]["y"] == b["series"][0]["y"]      # re-render is identical
    # the untouched value rides along for the tooltip and stays in {0, 1}
    assert set(a["series"][0]["customdata"]) <= {0.0, 1.0}


def test_scatter_adds_a_decile_trend_line():
    from evaluator.webapi.chart_data import wer_vs_recall_scatter

    spec = wer_vs_recall_scatter(_traces(50))
    trend = [s for s in spec["series"] if s["name"] == "mean per decile"]
    assert trend and trend[0]["mode"] == "markers+lines"
    assert spec["table"]["rows"]                           # the trend is the table twin


def test_scatter_reports_the_runs_own_correlation():
    from evaluator.webapi.chart_data import wer_vs_recall_scatter

    spec = wer_vs_recall_scatter(_traces(20), {"wer_recall5_correlation": -0.42})
    assert "-0.42" in spec["note"] or "−0.42" in spec["note"]


def test_score_margin_flags_negative_margins():
    """A distractor outranking every relevant doc is a different outcome, not a smaller
    number — it must be visible in the note and colored as a failure."""
    from evaluator.webapi.chart_data import score_margin_histogram

    losing = [{"relevant": {"d1": 1},
               "retrieved": [{"doc_key": "d2", "score": 0.9},
                             {"doc_key": "d1", "score": 0.1}]}]
    spec = score_margin_histogram(losing * 3, bins=4)
    assert "3 of 3" in spec["note"]
    assert "neg" in spec["series"][0]["colorkey"]


def test_score_margin_needs_both_a_hit_and_a_miss():
    from evaluator.webapi.chart_data import score_margin_histogram

    only_relevant = [{"relevant": {"d1": 1},
                      "retrieved": [{"doc_key": "d1", "score": 0.9}]}]
    assert score_margin_histogram(only_relevant)["empty"]


def test_per_query_specs_stay_json_safe_and_colorless():
    from evaluator.webapi import chart_data as C

    for spec in (C.per_query_histogram(_traces()),
                 C.wer_vs_recall_scatter(_traces()),
                 C.score_margin_histogram(_traces())):
        assert "#" not in json.dumps(spec)


# ── V7: efficiency + cross-run ───────────────────────────────────────────────

_PROV = {
    "timing": {"asr_s": 30.0, "embedding_s": 5.0, "retrieval_s": 1.0, "metrics_s": 0.0},
    "cache": {"asr": {"features": {"hits": 8, "misses": 2},
                      "transcriptions": {"hits": 0, "misses": 10}},
              "text_embedding": {"hits": 5, "misses": 5, "hit_rate": 0.5}},
    "cost": {"by_component": {"judge": {"prompt_tokens": 800, "completion_tokens": 200,
                                        "calls": 4}},
             "totals": {"total_tokens": 1000}, "budget_tokens": 4000},
}


def test_timing_sorted_and_shares_sum_to_one_hundred():
    from evaluator.webapi.chart_data import stage_timing

    spec = stage_timing(_PROV)
    assert spec["series"][0]["y"][-1] == "asr"        # longest ends up on top
    assert 0 not in spec["series"][0]["x"]            # zero-time stages are dropped
    total = sum(float(r[2].rstrip("%")) for r in spec["table"]["rows"])
    assert abs(total - 100.0) < 0.05


def test_timing_never_shares_an_axis_with_a_rate():
    """Dual axes invent relationships; cache reuse is a separate panel by design."""
    from evaluator.webapi.chart_data import stage_timing

    layout = stage_timing(_PROV)["layout"]
    assert "xaxis2" not in layout and "yaxis2" not in layout
    for trace in stage_timing(_PROV)["series"]:
        assert "xaxis" not in trace and "yaxis" not in trace


def test_timing_falls_back_to_top_level_latency():
    from evaluator.webapi.chart_data import stage_timing

    spec = stage_timing({}, {"asr_s": 2.0, "total_s": 9.9})
    assert spec["series"][0]["y"] == ["asr"]          # total_s is not a stage
    assert spec["series"][0]["x"] == [2.0]


def test_cache_flattens_nested_stages_and_derives_missing_rate():
    from evaluator.webapi.chart_data import cache_hit_rates

    spec = cache_hit_rates(_PROV)
    labels = spec["series"][0]["y"]
    assert "asr/features" in labels and "text_embedding" in labels
    rate = dict(zip(labels, spec["series"][0]["x"]))
    assert rate["asr/features"] == 0.8                # derived from hits/misses
    assert rate["text_embedding"] == 0.5              # taken from the reported hit_rate
    assert spec["layout"]["xaxis"]["range"] == [0, 1]


def test_token_budget_stacks_and_marks_the_budget():
    from evaluator.webapi.chart_data import token_budget

    spec = token_budget(_PROV)
    assert spec["layout"]["barmode"] == "stack"
    assert [s["name"] for s in spec["series"]] == ["prompt", "completion"]
    assert spec["layout"]["shapes"][0]["x0"] == 4000  # the budget is a rule, not a bar
    assert "25% used" in spec["note"]


def test_token_budget_without_a_budget_omits_the_rule():
    from evaluator.webapi.chart_data import token_budget

    spec = token_budget({"cost": {"by_component": {"judge": {"prompt_tokens": 5}}}})
    assert "shapes" not in spec["layout"]


def test_efficiency_panels_explain_themselves_when_absent():
    from evaluator.webapi.chart_data import cache_hit_rates, stage_timing, token_budget

    assert stage_timing({})["empty"]
    assert cache_hit_rates({})["empty"]
    assert token_budget({})["empty"]


_ROWS = [
    {"run_id": 1, "experiment_name": "a", "metric_value": 0.5,
     "duration_seconds": 90.0, "created_at": "2026-01-02"},
    {"run_id": 2, "experiment_name": "b", "metric_value": 0.7,
     "duration_seconds": 30.0, "created_at": "2026-01-01"},
    {"run_id": 3, "experiment_name": "c", "metric_value": None,
     "duration_seconds": 10.0, "created_at": "2026-01-03"},
]


def test_leaderboard_scatter_sorts_and_drops_incomplete_rows():
    from evaluator.webapi.chart_data import leaderboard_scatter

    spec = leaderboard_scatter(_ROWS, "MRR", "duration_seconds")
    assert spec["series"][0]["x"] == [30.0, 90.0]     # sorted, run 3 has no metric
    assert spec["series"][0]["mode"] == "markers"     # no line: duration is not a sequence


def test_leaderboard_over_time_is_chronological_and_connected():
    from evaluator.webapi.chart_data import leaderboard_scatter

    spec = leaderboard_scatter(_ROWS, "MRR", "created_at")
    assert spec["series"][0]["x"] == ["2026-01-01", "2026-01-02"]
    assert spec["series"][0]["mode"] == "markers+lines"
    assert spec["id"] == "metric-over-time"


# ── V8: compare page ─────────────────────────────────────────────────────────

_CMP = [
    {"name": "MRR", "a": 0.50, "b": 0.60},          # +20%
    {"name": "latency_ms", "a": 100.0, "b": 40.0},  # -60%
    {"name": "WER", "a": 0.20, "b": 0.21},          # +5%
    {"name": "zero_base", "a": 0.0, "b": 0.3},      # undefined → dropped
    {"name": "half_missing", "a": 0.4, "b": None},  # incomplete → dropped
]


def test_compare_uses_relative_change_so_scales_do_not_dominate():
    """Raw deltas would let a 60ms latency swing dwarf a 0.1 MRR gain."""
    from evaluator.webapi.chart_data import compare_relative_deltas

    spec = compare_relative_deltas(_CMP)
    by_name = dict(zip(spec["series"][0]["y"], spec["series"][0]["x"]))
    assert by_name["MRR"] == pytest.approx(20.0)
    assert by_name["latency_ms"] == pytest.approx(-60.0)
    assert spec["layout"]["xaxis"]["ticksuffix"] == "%"


def test_compare_orders_by_magnitude_of_change():
    from evaluator.webapi.chart_data import compare_relative_deltas

    spec = compare_relative_deltas(_CMP)
    # plotted bottom-up, so the largest mover is the LAST entry
    assert spec["series"][0]["y"][-1] == "latency_ms"


def test_compare_drops_undefined_and_incomplete_rows():
    from evaluator.webapi.chart_data import compare_relative_deltas

    plotted = set(compare_relative_deltas(_CMP)["series"][0]["y"])
    assert "zero_base" not in plotted        # relative change against 0 is undefined
    assert "half_missing" not in plotted


def test_compare_colors_by_direction():
    from evaluator.webapi.chart_data import compare_relative_deltas

    spec = compare_relative_deltas(_CMP)
    keys = dict(zip(spec["series"][0]["y"], spec["series"][0]["colorkey"]))
    assert keys["MRR"] == "pos" and keys["latency_ms"] == "neg"


def test_compare_states_that_it_is_unpaired():
    """Two independent runs have no CI and no significance test; the chart must say so
    rather than inviting the reader to infer one."""
    from evaluator.webapi.chart_data import compare_relative_deltas

    note = compare_relative_deltas(_CMP)["note"]
    assert "Unpaired" in note and "significance" in note
    assert all("error_x" not in s and "error_y" not in s
               for s in compare_relative_deltas(_CMP)["series"])


def test_compare_caps_the_bars_but_keeps_every_row_in_the_table():
    from evaluator.webapi.chart_data import compare_relative_deltas

    many = [{"name": f"m{i}", "a": 1.0, "b": 1.0 + i / 100.0} for i in range(30)]
    spec = compare_relative_deltas(many, top_n=5)
    assert len(spec["series"][0]["y"]) == 5
    # every row stays in the table, including m0's 0% change — only the BARS are capped
    assert len(spec["table"]["rows"]) == 30
    assert "of 30" in spec["note"]


def test_compare_without_shared_metrics_explains_itself():
    from evaluator.webapi.chart_data import compare_relative_deltas

    assert compare_relative_deltas([])["empty"]


def test_over_time_needs_two_runs_before_it_draws_a_trend():
    """One point gives Plotly a zero-width time range, which it pads to milliseconds and
    labels 22:24:43.999 … 22:24:44.001 — a chart that looks broken rather than empty."""
    from evaluator.webapi.chart_data import leaderboard_scatter

    one = [{"run_id": 1, "experiment_name": "a", "metric_value": 0.9,
            "created_at": "2026-01-01"}]
    spec = leaderboard_scatter(one, "MRR", "created_at")
    assert spec["empty"] and "two runs" in spec["empty"]
    # vs duration, a single point is a legitimate (if lonely) scatter
    one[0]["duration_seconds"] = 30.0
    assert not leaderboard_scatter(one, "MRR", "duration_seconds").get("empty")


def test_empty_state_copy_names_things_the_reader_recognises():
    from evaluator.webapi.chart_data import leaderboard_scatter

    msg = leaderboard_scatter([], "MRR", "duration_seconds")["empty"]
    assert "duration_seconds" not in msg and "run duration" in msg


def test_cache_bars_are_labelled_so_a_zero_row_is_not_an_invisible_bar():
    from evaluator.webapi.chart_data import cache_hit_rates

    spec = cache_hit_rates({"cache": {"asr": {"hits": 0, "misses": 4},
                                      "text": {"hits": 3, "misses": 1}}})
    assert spec["series"][0]["text"] == ["0%", "75%"]


# ── metric chips: the same number is stored up to three times ─────────────────

_DUPED = {
    "MRR": 0.9266666, "WER": 0.3508825, "retrieval_failure_rate": 0.0,
    "asr/mrr": 0.9266666,          # same number, branch-prefixed
    "asr/wer": 0.3508825,
    "asr/precision@1": 0.9,        # only exists namespaced — must survive
    "phased": True, "oracle_mode": False,
}


def test_branch_prefixed_repeats_are_dropped():
    from evaluator.webapi.chart_data import metric_chips

    labels = [c["label"] for c in metric_chips(_DUPED)["chips"]]
    assert "MRR" in labels and "asr/mrr" not in labels
    assert "asr/precision@1" in labels        # no unprefixed twin, so it is not a repeat


def test_a_branch_that_disagrees_keeps_both_chips():
    """Deduping on the name alone would hide a real per-branch difference."""
    from evaluator.webapi.chart_data import metric_chips

    labels = [c["label"] for c in metric_chips({"MRR": 0.5, "a/mrr": 0.5, "b/mrr": 0.7})["chips"]]
    assert labels == ["MRR", "b/mrr"]


def test_flags_are_not_printed_as_measurements():
    from evaluator.webapi.chart_data import metric_chips

    text = {c["label"]: c["text"] for c in metric_chips(_DUPED)["chips"]}
    assert text["phased"] == "yes" and text["oracle_mode"] == "no"


def test_confidence_intervals_ride_on_their_metric_and_never_get_a_chip():
    from evaluator.webapi.chart_data import metric_chips

    chips = metric_chips({"MRR": 0.9, "MRR_ci": [0.8, 0.95]})["chips"]
    assert len(chips) == 1 and chips[0]["ci"] == [0.8, 0.95]


def test_single_branch_values_add_nothing_but_a_disagreeing_branch_does():
    from evaluator.webapi.chart_data import metric_chips

    branches = {"asr": {"mrr": {"mean": 0.9266666}, "wer": {"mean": 0.3508825}}}
    assert metric_chips(_DUPED, branches)["branch_values"] is False
    moved = {"asr": {"mrr": {"mean": 0.4}}}
    assert metric_chips(_DUPED, moved)["branch_values"] is True
