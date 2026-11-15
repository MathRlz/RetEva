"""McNemar/Holm known values, per-query item_scores persisted in the report, and
scripts/paired_stats.py end to end on two synthetic result files."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from evaluator.analysis.significance import holm, mcnemar_exact
from evaluator.evaluation.aggregate import build_report
from evaluator.evaluation.item_set import ItemSet

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "paired_stats.py"
_spec = importlib.util.spec_from_file_location("paired_stats", _SCRIPT)
paired_stats = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(paired_stats)


def test_mcnemar_exact_known_value():
    # 8 A-only hits, 2 B-only hits, 5 concordant: p = 2 * P(X >= 8 | n=10, 0.5) = 112/1024
    a = [True] * 8 + [False] * 2 + [True] * 3 + [False] * 2
    b = [False] * 8 + [True] * 2 + [True] * 3 + [False] * 2
    assert mcnemar_exact(a, b) == (8, 2, pytest.approx(112 / 1024))
    assert mcnemar_exact([True, False], [True, False])[2] == 1.0


def test_holm_known_values():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm([0.5, 0.9]) == pytest.approx([1.0, 1.0])


def test_build_report_persists_item_scores():
    rep = build_report({"main": {"recall@1": ItemSet(["q1", "q2"], [1.0, 0.0])}})
    assert rep["item_scores"] == {"main": {"recall@1": {"q1": 1.0, "q2": 0.0}}}
    json.dumps(rep)  # must stay serializable


def _write(tmp_path, name, r1, mrr):
    ids = [f"q{i}" for i in range(len(r1))]
    report = {"item_scores": {"br": {
        "recall@1": dict(zip(ids, r1)), "mrr": dict(zip(ids, mrr)),
    }}}
    p = tmp_path / name
    p.write_text(json.dumps({"variants": {"v": {"report": report}}}))
    return str(p)


def test_script_end_to_end(tmp_path, monkeypatch, capsys):
    a = _write(tmp_path, "a.json", [1, 1, 0, 0] * 10, [1, 1, 0.5, 0.25] * 10)
    b = _write(tmp_path, "b.json", [1, 0, 0, 0] * 10, [1, 0.5, 0.5, 0.25] * 10)
    out_csv = tmp_path / "s.csv"
    monkeypatch.setattr(sys, "argv", [
        "paired_stats", "--pair", "a vs b", a, b + "@v/br", "--n-boot", "500", "--csv", str(out_csv),
    ])
    paired_stats.main()
    table = capsys.readouterr().out
    assert "| a vs b | 40 |" in table
    row = out_csv.read_text().splitlines()[1]
    assert ",10,0," in row  # b = 10 A-only hits, c = 0
    r = paired_stats.compare(paired_stats.load_scores(a), paired_stats.load_scores(b), 500)
    assert r["delta_recall@1"] == pytest.approx(-0.25)
    lo, hi = r["delta_recall@1_ci"]
    assert lo <= -0.25 <= hi and hi < 0
