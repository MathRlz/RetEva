"""E3: export_to_excel decomposed into summary + per-sample sheet writers.

Skipped where openpyxl is absent; otherwise round-trips a small results dict and reads the
cells back to pin the Summary metrics + the Per-Sample grid.
"""

import pytest

openpyxl = pytest.importorskip("openpyxl")

from evaluator.analysis.export import export_to_excel  # noqa: E402


def test_excel_export_writes_summary_and_per_sample(tmp_path):
    results = {
        "pipeline_mode": "asr_text_retrieval",
        "MRR": 0.5,
        "Recall@5": 0.75,
        "config": {"ignored": True},  # skip_keys → not a metric row
        "per_sample": {"MRR": [1.0, 0.0], "Recall@5": [1.0, 0.5]},
    }
    out = tmp_path / "r.xlsx"
    export_to_excel(results, str(out))

    wb = openpyxl.load_workbook(out)
    assert wb.sheetnames == ["Summary", "Per-Sample"]

    summary = {wb["Summary"][f"A{r}"].value: wb["Summary"][f"B{r}"].value
               for r in range(2, 12) if wb["Summary"][f"A{r}"].value}
    assert summary["pipeline_mode"] == "asr_text_retrieval"
    assert summary["MRR"] == 0.5
    assert summary["Recall@5"] == 0.75

    ps = wb["Per-Sample"]
    assert ps["A1"].value == "Sample"
    # two sample rows present
    assert ps["A2"].value == 1 and ps["A3"].value == 2


def test_excel_export_details_branch(tmp_path):
    results = {"MRR": 1.0, "details": [{"q": "a", "score": 0.1}, {"q": "b", "score": 0.2}]}
    out = tmp_path / "d.xlsx"
    export_to_excel(results, str(out))
    wb = openpyxl.load_workbook(out)
    assert "Per-Sample" in wb.sheetnames
    headers = [wb["Per-Sample"].cell(row=1, column=c).value for c in (1, 2)]
    assert headers == ["q", "score"]
