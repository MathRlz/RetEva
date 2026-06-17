"""Thesis-grade outputs from a branched run report (REPO_REVIEW batch 3).

A branched run's ``results["report"]`` block (built by ``evaluation.aggregate``)
carries per-branch means/CIs and paired cross-branch deltas with bootstrap CIs,
Wilcoxon p-values and BH-FDR q-values. This module turns that block into the
three publication artifacts:

- :func:`latex_branch_table` — branches as columns, metrics as rows, plus a delta
  block with CI + significance stars (the thesis "Table X" generator).
- :func:`plot_branch_deltas` — matplotlib bar chart of Δmetric per branch pair
  with CI whiskers and significance annotation.
- :func:`export_per_query_failures` — per-query CSV (WER / Recall@5 / retrieved
  docs / failure category) from ``query_traces`` for the failure narrative.

CLI: ``python -m evaluator.analysis.branch_report results_*.json --out-dir out/``
writes all applicable artifacts next to each other.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..logging_config import get_logger
from ._common import _escape_latex

logger = get_logger(__name__)

#: significance-star thresholds applied to the BH-FDR q-value (fall back to raw p).
_STAR_LEVELS = ((0.001, "***"), (0.01, "**"), (0.05, "*"))


def _stars(delta: Mapping[str, Any]) -> str:
    q = delta.get("p_value_fdr", delta.get("p_value"))
    if not isinstance(q, (int, float)):
        return ""
    for threshold, mark in _STAR_LEVELS:
        if q < threshold:
            return mark
    return ""


def _branch_metric_names(report: Mapping[str, Any]) -> List[str]:
    names: List[str] = []
    for metrics in (report.get("branches") or {}).values():
        for name in metrics:
            if name not in names:
                names.append(name)
    return names


def latex_branch_table(
    report: Mapping[str, Any],
    *,
    metrics: Optional[Sequence[str]] = None,
    caption: str = "Per-branch results with paired deltas",
    label: str = "tab:branch-results",
) -> str:
    """Render ``report.branches`` + ``report.deltas`` as a LaTeX ``table``.

    Branches are columns, metrics rows (mean, with 95% CI when present). A second
    block lists each paired delta with its bootstrap CI and significance stars on
    the BH-FDR q-value (*** q<.001, ** q<.01, * q<.05). ``metrics`` restricts and
    orders the rows; default = every metric any branch reports.
    """
    branches: Dict[str, Any] = dict(report.get("branches") or {})
    if not branches:
        raise ValueError("report has no 'branches' block — nothing to tabulate")
    metric_names = list(metrics) if metrics else _branch_metric_names(report)
    branch_names = list(branches)

    cols = "l" + "c" * len(branch_names)
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        rf"\caption{{{_escape_latex(caption)}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{cols}}}",
        r"\toprule",
        "Metric & " + " & ".join(_escape_latex(b) for b in branch_names) + r" \\",
        r"\midrule",
    ]
    for name in metric_names:
        cells = []
        for b in branch_names:
            stats = branches[b].get(name)
            if not isinstance(stats, Mapping) or "mean" not in stats:
                cells.append("--")
                continue
            cell = f"{stats['mean']:.4f}"
            ci = stats.get("ci")
            if isinstance(ci, (list, tuple)) and len(ci) == 2:
                cell += f" [{ci[0]:.3f}, {ci[1]:.3f}]"
            cells.append(cell)
        lines.append(f"{_escape_latex(name)} & " + " & ".join(cells) + r" \\")

    deltas: Dict[str, Any] = dict(report.get("deltas") or {})
    if deltas:
        lines += [
            r"\midrule",
            r"\multicolumn{%d}{l}{\emph{Paired deltas}} \\" % (len(branch_names) + 1),
        ]
        for pair, dmetrics in deltas.items():
            for name in metric_names:
                d = dmetrics.get(name)
                if not isinstance(d, Mapping) or "mean_delta" not in d:
                    continue
                cell = f"{d['mean_delta']:+.4f}{_stars(d)}"
                ci = d.get("ci")
                if isinstance(ci, (list, tuple)) and len(ci) == 2:
                    cell += f" [{ci[0]:.3f}, {ci[1]:.3f}]"
                cell += f" (n={d.get('n_paired', '?')})"
                row_label = _escape_latex(f"Δ{name} ({pair})")
                lines.append(
                    rf"{row_label} & \multicolumn{{{len(branch_names)}}}{{c}}{{{cell}}} \\"
                )
        lines.append(
            r"\multicolumn{%d}{l}{\footnotesize Stars: BH-FDR q-value "
            r"(*** $q<.001$, ** $q<.01$, * $q<.05$); brackets: 95\%% bootstrap CI.} \\"
            % (len(branch_names) + 1)
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def plot_branch_deltas(
    report: Mapping[str, Any],
    output_path: str,
    *,
    metrics: Optional[Sequence[str]] = None,
    title: str = "Paired per-branch deltas (95% bootstrap CI)",
) -> str:
    """Bar chart of every paired delta with CI whiskers; stars mark FDR significance.

    One bar per (branch pair, metric). Writes ``output_path`` (format from the
    extension; PNG/PDF both fine) and returns it.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    deltas: Dict[str, Any] = dict(report.get("deltas") or {})
    if not deltas:
        raise ValueError("report has no 'deltas' block — nothing to plot")
    metric_names = list(metrics) if metrics else None

    labels: List[str] = []
    values: List[float] = []
    err_lo: List[float] = []
    err_hi: List[float] = []
    stars: List[str] = []
    for pair, dmetrics in deltas.items():
        for name, d in dmetrics.items():
            if metric_names is not None and name not in metric_names:
                continue
            if not isinstance(d, Mapping) or "mean_delta" not in d:
                continue
            v = float(d["mean_delta"])
            labels.append(f"Δ{name}\n{pair}")
            values.append(v)
            ci = d.get("ci")
            if isinstance(ci, (list, tuple)) and len(ci) == 2:
                err_lo.append(max(0.0, v - float(ci[0])))
                err_hi.append(max(0.0, float(ci[1]) - v))
            else:
                err_lo.append(0.0)
                err_hi.append(0.0)
            stars.append(_stars(d))
    if not labels:
        raise ValueError("no delta entries matched the requested metrics")

    fig, ax = plt.subplots(figsize=(max(6.0, 1.4 * len(labels)), 4.5))
    x = range(len(labels))
    colors = ["#2563eb" if v >= 0 else "#dc2626" for v in values]
    ax.bar(x, values, yerr=[err_lo, err_hi], capsize=4, color=colors, alpha=0.85)
    ax.axhline(0.0, color="#374151", linewidth=0.8)
    for xi, (v, hi, mark) in enumerate(zip(values, err_hi, stars)):
        if mark:
            ax.annotate(
                mark, (xi, v + hi), textcoords="offset points", xytext=(0, 4),
                ha="center", fontsize=12, fontweight="bold",
            )
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Δ (branch − baseline)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    logger.info("delta plot written to %s", output_path)
    return output_path


_FAILURE_FIELDS = [
    "query_id", "question", "per_query_wer", "per_query_cer", "recall_at_5",
    "failure_category", "n_relevant", "top_doc_key", "top_score",
    "relevant_rank", "generated_answer", "reference_answer",
]


def _categorize(trace: Mapping[str, Any]) -> Dict[str, Any]:
    """Per-query failure category from a trace entry.

    ``hit`` — a relevant doc is in the retrieved list; ``near_miss`` — relevant doc
    retrieved but below the recall cut (rank > 5); ``retrieval_miss`` — no relevant
    doc retrieved at all; ``no_judgment`` — the trace carries no relevance info.
    """
    relevant = trace.get("relevant") or {}
    retrieved = trace.get("retrieved") or []
    if not relevant:
        return {"failure_category": "no_judgment", "relevant_rank": None}
    for rank, doc in enumerate(retrieved, start=1):
        if doc.get("doc_key") in relevant:
            return {
                "failure_category": "hit" if rank <= 5 else "near_miss",
                "relevant_rank": rank,
            }
    return {"failure_category": "retrieval_miss", "relevant_rank": None}


def _query_traces(results: Mapping[str, Any]) -> Any:
    """Per-query traces, G5-canonical under ``report['traces']`` (top-level fallback for
    result files written before the cutover)."""
    report = results.get("report") or {}
    return report.get("traces", {}).get("query_traces") or results.get("query_traces") or []


def export_per_query_failures(results: Mapping[str, Any], output_path: str) -> str:
    """Write the per-query failure table (CSV) from ``report['traces']['query_traces']``.

    Needs a run with ``trace_limit > 0`` (traces carry per-query WER/recall and the
    retrieved list); raises otherwise so a missing-trace run can't silently produce
    an empty table.
    """
    traces = _query_traces(results)
    if not traces:
        raise ValueError(
            "results carry no 'query_traces' — rerun with evaluation.trace_limit > 0 "
            "to export per-query failures"
        )
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_FAILURE_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for trace in traces:
            row = {k: trace.get(k) for k in _FAILURE_FIELDS}
            row["n_relevant"] = len(trace.get("relevant") or {})
            retrieved = trace.get("retrieved") or []
            if retrieved:
                row["top_doc_key"] = retrieved[0].get("doc_key")
                row["top_score"] = retrieved[0].get("score")
            row.update(_categorize(trace))
            writer.writerow(row)
    logger.info("per-query failure table written to %s (%d rows)", output_path, len(traces))
    return output_path


def export_all(results_path: str, out_dir: str = ".") -> Dict[str, str]:
    """Write every applicable artifact for one results JSON; returns {kind: path}.

    Skips (with a log line) artifacts whose inputs the run didn't produce —
    e.g. no ``report.deltas`` on a single-branch run, no traces without
    ``trace_limit`` — instead of failing the whole export.
    """
    results = json.loads(Path(results_path).read_text())
    report = results.get("report") or {}
    stem = Path(results_path).stem
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: Dict[str, str] = {}
    if report.get("branches"):
        path = str(out / f"{stem}_branch_table.tex")
        Path(path).write_text(latex_branch_table(report))
        written["latex"] = path
    else:
        logger.info("skip latex table: no report.branches in %s", results_path)
    if report.get("deltas"):
        written["plot"] = plot_branch_deltas(report, str(out / f"{stem}_deltas.png"))
    else:
        logger.info("skip delta plot: no report.deltas in %s", results_path)
    if _query_traces(results):
        written["failures"] = export_per_query_failures(
            results, str(out / f"{stem}_failures.csv")
        )
    else:
        logger.info("skip failure csv: no query_traces in %s", results_path)
    return written


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Export thesis artifacts (LaTeX branch table, delta plot, "
        "per-query failure CSV) from evaluator results JSON files."
    )
    parser.add_argument("results", nargs="+", help="results_*.json file(s)")
    parser.add_argument("--out-dir", default=".", help="output directory")
    args = parser.parse_args(argv)
    for results_path in args.results:
        written = export_all(results_path, args.out_dir)
        for kind, path in written.items():
            print(f"{results_path}: {kind} -> {path}")
        if not written:
            print(f"{results_path}: nothing exportable (no report/traces)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
