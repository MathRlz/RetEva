"""Tidy, stable exports of an evaluation report (architecture-improvements §6).

Two researcher-facing shapes the deeply-nested report doesn't give directly:

* :func:`report_to_metrics_table` — one row per ``(branch, metric)`` with mean / CI / n,
  the flat table leaderboards, dataframes, and paper tables actually want.
* :func:`report_query_traces` — the per-query trace records (query, retrieved keys + scores,
  relevance, judge verdict, per-query metrics) for downstream ML / error analysis.

Writers cover CSV + JSONL (no extra deps) and optional Parquet (pandas + pyarrow).
"""

from __future__ import annotations

import csv
import json
from typing import Any, Dict, List

_METRICS_COLUMNS = ["branch", "metric", "mean", "ci_lower", "ci_upper", "n"]


def _report_block(report: Dict[str, Any]) -> Dict[str, Any]:
    """Accept either the full results dict (with a ``report`` key) or the report block."""
    if isinstance(report, dict):
        if "branches" in report or "traces" in report:
            return report
        inner = report.get("report")
        if isinstance(inner, dict):
            return inner
    return report or {}


def report_to_metrics_table(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten ``report.branches`` into ``[{branch, metric, mean, ci_lower, ci_upper, n}, …]``.

    Rows are ordered by branch then metric (sorted) for a stable, diff-able table.
    """
    block = _report_block(report)
    rows: List[Dict[str, Any]] = []
    for branch in sorted((block.get("branches") or {})):
        metrics = block["branches"][branch] or {}
        for metric in sorted(metrics):
            val = metrics[metric]
            if isinstance(val, dict):
                rows.append({
                    "branch": branch,
                    "metric": metric,
                    "mean": val.get("mean"),
                    "ci_lower": val.get("ci_lower"),
                    "ci_upper": val.get("ci_upper"),
                    "n": val.get("n"),
                })
            elif isinstance(val, (int, float)) and not isinstance(val, bool):
                rows.append({
                    "branch": branch, "metric": metric, "mean": val,
                    "ci_lower": None, "ci_upper": None, "n": None,
                })
    return rows


def report_query_traces(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The per-query trace records (``report.traces.query_traces``), or ``[]``."""
    traces = _report_block(report).get("traces") or {}
    if isinstance(traces, dict):
        return list(traces.get("query_traces") or [])
    if isinstance(traces, list):
        return traces
    return []


def write_metrics_table_csv(report: Dict[str, Any], path: str) -> int:
    """Write the metrics table to a CSV; return the row count."""
    rows = report_to_metrics_table(report)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_METRICS_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def write_traces_jsonl(report: Dict[str, Any], path: str) -> int:
    """Write per-query traces as JSON Lines (one trace per line); return the count."""
    rows = report_query_traces(report)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, default=str) + "\n")
    return len(rows)


def write_traces_parquet(report: Dict[str, Any], path: str) -> int:
    """Write per-query traces as Parquet (needs pandas + pyarrow); return the count."""
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - optional dep
        raise ImportError(
            "Parquet export needs pandas + pyarrow (pip install pandas pyarrow)"
        ) from exc
    rows = report_query_traces(report)
    pd.DataFrame(rows).to_parquet(path)
    return len(rows)
