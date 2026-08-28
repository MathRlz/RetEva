"""V3 pins: a shaped spec reaches the page as parseable JSON with its table twin.

Guards the seam between the pure shaping (test_chart_data.py) and the template macro:
the spec must ship inside <script type="application/json"> (so untrusted labels are never
concatenated into executable JS) and it must be strict JSON — bare NaN would make
JSON.parse throw and silently blank the panel.
"""

import json
import re

from evaluator.storage import ExperimentStore

_REPORT = {
    "branches": {"asr": {"recall@5": {"mean": 0.6, "n": 200}},
                 "ref": {"recall@5": {"mean": 0.8, "n": 200}}},
    "deltas": {"asr_vs_ref": {"recall@5": {
        "mean_delta": -0.2, "ci": [-0.25, -0.15], "cohens_d": -0.53,
        "p_value_fdr": 1e-9, "n_paired": 200, "n_only_branch": 12,
        "n_only_baseline": 0, "drop_biased": True,
    }}},
}


def _seed(tmp_path, report):
    store = ExperimentStore(db_path=str(tmp_path / "leaderboard.sqlite"))
    return store.upsert_run(
        experiment_name="exp", dataset_name="pubmed_qa",
        pipeline_mode="asr_text_retrieval", duration_seconds=1.0,
        config={"experiment_name": "exp"},
        metrics={"MRR": 0.5, "report": report},
    )


def _specs(html):
    return {m[0]: json.loads(m[1]) for m in re.findall(
        r'<script type="application/json" id="([^"]+)-spec">(.*?)</script>', html, re.S)}


def test_panels_ship_parseable_specs_with_tables(client, tmp_path):
    rid = _seed(tmp_path, _REPORT)
    html = client.get(f"/ui/runs/{rid}?output_dir={tmp_path}").text

    specs = _specs(html)                      # json.loads == JSON.parse's strictness
    assert "delta-forest" in specs and "delta-denominators" in specs
    forest = specs["delta-forest"]
    assert forest["series"][0]["x"] == [-0.2]
    assert "#" not in json.dumps(forest)      # colors stay semantic keys
    # the accessible twin is rendered server-side from the same spec
    # (Jinja escapes the apostrophe in the "Cohen's d" column header)
    assert "Data table" in html and "Cohen" in html
    # drop-bias is surfaced, not buried
    assert "⚠" in html


def test_single_branch_run_explains_itself(client, tmp_path):
    """A run with no deltas must render a sentence, not an empty box."""
    rid = _seed(tmp_path, {"branches": {"main": {"mrr": {"mean": 0.4, "n": 5}}}})
    html = client.get(f"/ui/runs/{rid}?output_dir={tmp_path}").text
    assert "single-branch run" in html
    assert not _specs(html)                   # nothing to plot → no spec emitted


def test_degenerate_ci_does_not_break_the_panel(client, tmp_path):
    """n=1 / zero variance produces NaN bounds upstream; the page must still parse."""
    rid = _seed(tmp_path, {"deltas": {"a_vs_b": {"m": {
        "mean_delta": 0.0, "ci": [float("nan"), float("nan")],
        "cohens_d": float("nan"), "n_paired": 1,
    }}}})
    html = client.get(f"/ui/runs/{rid}?output_dir={tmp_path}").text
    assert "NaN" not in html
    assert _specs(html)["delta-forest"]["series"][0]["x"] == [0.0]
