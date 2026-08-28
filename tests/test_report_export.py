"""Roadmap §6 — tidy report exports: metrics table + per-query traces."""

import csv
import json

from evaluator.analysis.report_export import (
    report_query_traces,
    report_to_metrics_table,
    write_metrics_table_csv,
    write_traces_jsonl,
)

_REPORT = {
    "report": {
        "branches": {
            "asr_text_retrieval": {
                "mrr": {"mean": 0.8, "ci_lower": 0.7, "ci_upper": 0.9, "n": 50},
                "wer": {"mean": 0.12, "n": 50},
            },
        },
        "traces": {
            "query_traces": [
                {"query_id": "q1", "per_query_wer": 0.1, "recall_at_5": 1.0},
                {"query_id": "q2", "per_query_wer": 0.3, "recall_at_5": 0.0},
            ],
        },
        "provenance": {"seed": 42},
    }
}


def test_metrics_table_flattens_branches_sorted():
    rows = report_to_metrics_table(_REPORT)
    assert [(r["branch"], r["metric"]) for r in rows] == [
        ("asr_text_retrieval", "mrr"),
        ("asr_text_retrieval", "wer"),
    ]
    mrr = rows[0]
    assert mrr["mean"] == 0.8 and mrr["ci_lower"] == 0.7 and mrr["n"] == 50
    assert rows[1]["ci_lower"] is None  # wer had no CI


def test_accepts_report_block_or_full_results():
    assert report_to_metrics_table(_REPORT) == report_to_metrics_table(_REPORT["report"])


def test_query_traces_extracted():
    tr = report_query_traces(_REPORT)
    assert [t["query_id"] for t in tr] == ["q1", "q2"]


def test_write_metrics_csv(tmp_path):
    out = tmp_path / "m.csv"
    n = write_metrics_table_csv(_REPORT, str(out))
    assert n == 2
    rows = list(csv.DictReader(out.open()))
    assert rows[0]["metric"] == "mrr" and rows[0]["mean"] == "0.8"


def test_write_traces_jsonl(tmp_path):
    out = tmp_path / "t.jsonl"
    n = write_traces_jsonl(_REPORT, str(out))
    assert n == 2
    lines = [json.loads(line) for line in out.read_text().splitlines()]
    assert lines[1]["query_id"] == "q2"


def test_cli_export_metrics_table_and_traces(tmp_path):
    from evaluator.cli.export import main as export_main

    report_path = tmp_path / "results.json"
    report_path.write_text(json.dumps(_REPORT))

    mt = tmp_path / "table.csv"
    assert export_main([str(report_path), "-f", "metrics-table", "-o", str(mt)]) == 0
    assert mt.exists()

    tj = tmp_path / "tr.jsonl"
    assert export_main([str(report_path), "-f", "traces", "-o", str(tj)]) == 0
    assert len(tj.read_text().splitlines()) == 2
