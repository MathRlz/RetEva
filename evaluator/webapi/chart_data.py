"""Chart shaping: report/report-adjacent dicts → plain, renderable chart specs.

Every function here is **pure** — dicts in, dicts out, no FastAPI, no Jinja, no Plotly —
so the logic that decides what a chart says is unit-testable even though the rendering
(a thin JS layer) is not. Templates receive a spec and hand it to `plotFromSpec`; they
never compute.

The spec envelope::

    {id, title, note, empty, series[], layout{}, table{caption, columns, rows[]}}

* ``empty`` — a human sentence explaining why there is nothing to draw. Present instead
  of ``series`` whenever the inputs are missing; callers render the sentence rather than
  an empty box, and nothing raises on a metrics-poor run.
* ``series`` — near-verbatim Plotly traces, except colors are semantic ``colorkey``
  strings ("cat1", "pos", "ord") resolved against the theme tokens in JS. A hex here
  would freeze the chart in one theme.
* ``table`` — the same numbers the chart plots, rendered server-side as the accessible
  twin. It is built from the same arrays, so the table and the chart cannot disagree.

**Non-finite floats are mapped to None** by :func:`_num` on the way out. This is not
cosmetic: the spec ships inside ``<script type="application/json">`` and ``JSON.parse``
*rejects* bare ``NaN``/``Infinity`` (which both ``json.dumps`` and Jinja's ``|tojson``
emit), so a degenerate CI — n=1, zero variance — would otherwise kill the whole panel.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional

__all__ = [
    "chart_spec",
    "empty_spec",
    "delta_forest",
    "delta_volcano",
    "denominator_bars",
    "rank_distribution",
    "failure_categories",
    "recall_loss_by_k",
    "per_query_histogram",
    "wer_vs_recall_scatter",
    "score_margin_histogram",
    "stage_timing",
    "cache_hit_rates",
    "token_budget",
    "leaderboard_scatter",
    "compare_relative_deltas",
    "metric_chips",
]


def _num(value: Any) -> Optional[float]:
    """A JSON-safe float: NaN/Infinity → None (see the module docstring)."""
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _nums(values: Iterable[Any]) -> List[Optional[float]]:
    return [_num(v) for v in values]


def empty_spec(chart_id: str, title: str, reason: str) -> Dict[str, Any]:
    """A spec that renders as a sentence — the honest form of 'no chart here'."""
    return {"id": chart_id, "title": title, "note": "", "empty": reason,
            "series": [], "layout": {}, "table": None}


def chart_spec(
    chart_id: str,
    title: str,
    series: List[Dict[str, Any]],
    *,
    note: str = "",
    layout: Optional[Dict[str, Any]] = None,
    table: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {"id": chart_id, "title": title, "note": note, "empty": None,
            "series": series, "layout": layout or {}, "table": table}


def _iter_deltas(report: Dict[str, Any]):
    """``(pair, metric, delta_dict)`` for every delta carrying a numeric mean."""
    for pair, metrics in sorted((report.get("deltas") or {}).items()):
        if not isinstance(metrics, dict):
            continue
        for metric, delta in sorted(metrics.items()):
            if isinstance(delta, dict) and _num(delta.get("mean_delta")) is not None:
                yield pair, metric, delta


def _q_of(delta: Dict[str, Any]) -> Optional[float]:
    """The FDR-corrected q, falling back to the raw p (matching the CLI's star logic)."""
    q = _num(delta.get("p_value_fdr"))
    return q if q is not None else _num(delta.get("p_value"))


def delta_forest(report: Dict[str, Any], chart_id: str = "delta-forest") -> Dict[str, Any]:
    """Per-branch metric deltas as a forest plot: point estimate + 95% CI, ordered by
    effect size.

    A forest, not a bar chart: what the reader must judge is *where the interval sits
    relative to zero*, and a bar adds area-from-zero that isn't part of the claim and
    visually buries a CI that crosses it.
    """
    title = "Effect sizes with confidence intervals"
    rows = list(_iter_deltas(report))
    if not rows:
        return empty_spec(chart_id, title,
                          "This run has no cross-branch deltas (a single-branch run).")

    # Largest absolute effect first; deltas without a Cohen's d sort last rather than
    # being dropped — they still have a point estimate worth showing.
    def _sort_key(row):
        d = _num(row[2].get("cohens_d"))
        return (d is None, -abs(d) if d is not None else 0.0)

    rows.sort(key=_sort_key)

    labels = [f"Δ{metric} ({pair})" for pair, metric, _ in rows]
    values = [_num(d.get("mean_delta")) for _, _, d in rows]
    plus, minus, table_rows = [], [], []
    for (pair, metric, d), mean in zip(rows, values):
        ci = d.get("ci")
        lo = hi = None
        if isinstance(ci, (list, tuple)) and len(ci) == 2:
            lo, hi = _num(ci[0]), _num(ci[1])
        # Clamp: an inverted or asymmetric interval would draw a negative whisker.
        plus.append(max(0.0, hi - mean) if (hi is not None and mean is not None) else 0.0)
        minus.append(max(0.0, mean - lo) if (lo is not None and mean is not None) else 0.0)
        q = _q_of(d)
        table_rows.append([
            f"Δ{metric} ({pair})",
            mean,
            None if lo is None else lo,
            None if hi is None else hi,
            _num(d.get("cohens_d")),
            q,
            _num(d.get("n_paired")),
        ])

    series = [{
        "type": "scatter",
        "mode": "markers",
        "orientation": "h",
        "x": values,
        "y": labels,
        "colorkey": "cat1",
        "error_x": {"type": "data", "array": plus, "arrayminus": minus,
                    "colorkey": "muted"},
        "hovertemplate": "<b>%{x:.4f}</b><br>%{y}<extra></extra>",
        "name": "Δ vs baseline",
    }]
    layout = {
        "xaxis": {"title": "Δ (branch − baseline)"},
        "yaxis": {"automargin": True},
        # A rule at zero is the reference the whole chart is read against.
        "shapes": [{"type": "line", "x0": 0, "x1": 0, "yref": "paper", "y0": 0, "y1": 1,
                    "colorkey": "zero"}],
        "margin": {"l": 200, "r": 24, "t": 16, "b": 48},
    }
    return chart_spec(
        chart_id, title, series,
        note="Ordered by |Cohen's d|. Whiskers are 95% bootstrap CIs; an interval "
             "crossing zero means the direction is not established.",
        layout=layout,
        table={"caption": "Per-metric deltas with confidence intervals and effect sizes",
               "columns": ["Delta", "Mean", "CI low", "CI high", "Cohen's d", "q", "n paired"],
               "rows": table_rows},
    )


#: q below this counts as significant (matches aggregate.py's `significant` flag).
_ALPHA = 0.05


def delta_volcano(report: Dict[str, Any], chart_id: str = "delta-volcano") -> Dict[str, Any]:
    """Every delta at once: effect on x, evidence on y.

    Two continuous axes is the whole point — a metric can move a lot on weak evidence
    (right, low) or barely move with overwhelming evidence (centre, high), and neither
    reads off a bar chart. Significance is carried by symbol *and* color, never color
    alone, so it survives CVD and print.
    """
    title = "Effect vs evidence"
    rows = [(pair, metric, d) for pair, metric, d in _iter_deltas(report)
            if _q_of(d) is not None]
    if not rows:
        return empty_spec(chart_id, title,
                          "No p-values in this report — significance testing needs "
                          "paired branches.")

    groups: Dict[bool, Dict[str, List[Any]]] = {
        True: {"x": [], "y": [], "text": []},
        False: {"x": [], "y": [], "text": []},
    }
    table_rows = []
    for pair, metric, d in rows:
        q = _q_of(d)
        # q=0 underflows to inf on a log scale; clamp to the smallest float that plots.
        y = -math.log10(max(q, 1e-300))
        sig = bool(d.get("significant")) if d.get("significant") is not None else q < _ALPHA
        g = groups[sig]
        g["x"].append(_num(d.get("mean_delta")))
        g["y"].append(_num(y))
        g["text"].append(f"Δ{metric} ({pair})")
        table_rows.append([f"Δ{metric} ({pair})", _num(d.get("mean_delta")), q,
                           "yes" if sig else "no"])

    hover = "<b>%{text}</b><br>Δ %{x:.4f}<br>−log10 q %{y:.2f}<extra></extra>"
    series = []
    if groups[False]["x"]:
        series.append({"type": "scatter", "mode": "markers", "name": "not significant",
                       "x": groups[False]["x"], "y": groups[False]["y"],
                       "text": groups[False]["text"], "colorkey": "dim",
                       "marker": {"size": 11, "symbol": "x"}, "hovertemplate": hover})
    if groups[True]["x"]:
        series.append({"type": "scatter", "mode": "markers",
                       "name": f"q < {_ALPHA}", "x": groups[True]["x"],
                       "y": groups[True]["y"], "text": groups[True]["text"],
                       "colorkey": "cat1",
                       "marker": {"size": 12, "symbol": "circle"},
                       "hovertemplate": hover})

    layout = {
        "xaxis": {"title": "Δ (branch − baseline)"},
        "yaxis": {"title": "−log10(q)"},
        "shapes": [
            {"type": "line", "x0": 0, "x1": 0, "yref": "paper", "y0": 0, "y1": 1,
             "colorkey": "zero"},
            {"type": "line", "xref": "paper", "x0": 0, "x1": 1,
             "y0": -math.log10(_ALPHA), "y1": -math.log10(_ALPHA), "colorkey": "muted"},
        ],
        "legend": {"orientation": "h", "y": -0.25},
        "margin": {"l": 64, "r": 24, "t": 16, "b": 64},
    }
    return chart_spec(
        chart_id, title, series,
        note=f"Above the horizontal rule is q < {_ALPHA} (BH-FDR). Far from the vertical "
             f"rule is a large effect. Both matter: a tiny but certain change sits high "
             f"and centre.",
        layout=layout,
        table={"caption": "Effect size and corrected p-value per delta",
               "columns": ["Delta", "Mean Δ", "q", "Significant"],
               "rows": table_rows},
    )


def denominator_bars(report: Dict[str, Any],
                     chart_id: str = "delta-denominators") -> Dict[str, Any]:
    """How many items each paired comparison actually rests on.

    Stacked parts-of-a-whole because that is the literal claim: of the items in play,
    this many were compared pairwise and these were dropped from one side only. A
    comparison flagged ``drop_biased`` (>5% one-sided exclusions) is marked in its label
    — items rarely drop at random, so past that point the intersection measures the
    survivors rather than the branch.
    """
    title = "Paired sample sizes"
    rows = list(_iter_deltas(report))
    if not rows:
        return empty_spec(chart_id, title, "No paired comparisons in this run.")

    labels, paired, only_b, only_base, table_rows = [], [], [], [], []
    biased_any = False
    for pair, metric, d in rows:
        biased = bool(d.get("drop_biased"))
        biased_any = biased_any or biased
        labels.append(f"Δ{metric} ({pair})" + (" ⚠" if biased else ""))
        paired.append(_num(d.get("n_paired")) or 0)
        only_b.append(_num(d.get("n_only_branch")) or 0)
        only_base.append(_num(d.get("n_only_baseline")) or 0)
        table_rows.append([
            f"Δ{metric} ({pair})", paired[-1], only_b[-1], only_base[-1],
            "yes" if biased else "no", d.get("denominator_policy") or "",
        ])

    def _stack(name: str, values: List[Any], colorkey: str) -> Dict[str, Any]:
        return {"type": "bar", "orientation": "h", "name": name,
                "x": values, "y": labels, "colorkey": colorkey,
                "hovertemplate": "<b>%{x:.0f}</b> " + name + "<br>%{y}<extra></extra>"}

    series = [
        _stack("paired", paired, "cat1"),
        _stack("branch only", only_b, "cat2"),
        _stack("baseline only", only_base, "cat3"),
    ]
    note = ("Only the paired items enter the statistics. "
            + ("⚠ marks comparisons where one-sided drops exceed 5% — the surviving "
               "intersection may be easier than the full set." if biased_any else
               "No comparison here exceeded the 5% one-sided drop threshold."))
    return chart_spec(
        chart_id, title, series,
        note=note,
        layout={"barmode": "stack", "xaxis": {"title": "queries"},
                "margin": {"l": 200, "r": 24, "t": 16, "b": 48},
                "legend": {"orientation": "h", "y": -0.25}},
        table={"caption": "Paired and one-sided-only item counts per comparison",
               "columns": ["Delta", "Paired", "Branch only", "Baseline only",
                           "Drop biased", "Policy"],
               "rows": table_rows},
    )


#: Ranks are an ORDERED scale, so the buckets have a fixed order and an ordinal ramp —
#: never dict insertion order, and never categorical hues (swapping two buckets would
#: change the meaning).
RANK_BUCKETS = ("1", "2", "3-5", "6+", "not_found")


def rank_distribution(fail: Dict[str, Any],
                      chart_id: str = "rank-distribution") -> Dict[str, Any]:
    """Where the first relevant document landed.

    The single most diagnostic retrieval picture: a run with the same recall@5 can be
    "usually rank 1" or "usually rank 4", and only the shape says which.
    """
    title = "First relevant document — rank"
    ranks = (fail or {}).get("rank_distribution") or {}
    counts = [_num(ranks.get(b)) or 0 for b in RANK_BUCKETS]
    if not any(counts):
        return empty_spec(chart_id, title,
                          "No rank distribution recorded (needs retrieval with "
                          "ground-truth judgements).")
    total = sum(counts) or 1
    series = [{
        "type": "bar",
        "x": list(RANK_BUCKETS),
        "y": counts,
        # ordinal ramp: light→dark reads as "further down the ranking", and the
        # not-found bucket takes the failure color rather than the next ramp step.
        "colorkey": ["ord1", "ord2", "ord3", "ord4", "neg"],
        "hovertemplate": "<b>%{y:.0f}</b> queries at rank %{x}<extra></extra>",
    }]
    return chart_spec(
        chart_id, title, series,
        note="Lower buckets are better; 'not_found' means no relevant document was "
             "retrieved at all.",
        layout={"xaxis": {"title": "rank of first relevant doc"},
                "yaxis": {"title": "queries"}},
        table={"caption": "Queries by rank of the first relevant document",
               "columns": ["Rank", "Queries", "Share"],
               "rows": [[b, c, f"{100.0 * c / total:.1f}%"]
                        for b, c in zip(RANK_BUCKETS, counts)]},
    )


def failure_categories(fail: Dict[str, Any],
                       chart_id: str = "failure-categories") -> Dict[str, Any]:
    """Why retrieval missed, split by cause.

    Horizontal bars: the category names are long, and the reader is comparing a handful
    of magnitudes, not a composition — so one hue and a sorted axis beat a stack here.
    """
    title = "Failure causes"
    cats = (fail or {}).get("failure_categories") or {}
    pairs = [(str(k), _num(v) or 0) for k, v in cats.items() if (_num(v) or 0) > 0]
    if not pairs:
        return empty_spec(chart_id, title, "No categorized retrieval failures in this run.")
    pairs.sort(key=lambda kv: kv[1])          # largest at the top after the y reversal
    labels = [k.replace("_", " ") for k, _ in pairs]
    counts = [v for _, v in pairs]
    series = [{
        "type": "bar", "orientation": "h", "x": counts, "y": labels,
        "colorkey": "cat1",
        "hovertemplate": "<b>%{x:.0f}</b> queries<br>%{y}<extra></extra>",
    }]
    return chart_spec(
        chart_id, title, series,
        note="corpus gap = the relevant document is not in the corpus; asr failure = "
             "missed with high WER; embedding mismatch = missed despite clean text.",
        layout={"xaxis": {"title": "queries"},
                "margin": {"l": 160, "r": 24, "t": 16, "b": 48}},
        table={"caption": "Retrieval failures by cause",
               "columns": ["Cause", "Queries"],
               "rows": [[k, v] for k, v in reversed(pairs)]},
    )


def recall_loss_by_k(report: Dict[str, Any],
                     chart_id: str = "recall-loss") -> Dict[str, Any]:
    """How much recall each branch loses against the baseline, as the cutoff k grows.

    A line, because k is ordered and the question is where the loss saturates — that is
    the "which k should I ship" decision.
    """
    title = "Retrieval loss vs cutoff"
    impact = (report or {}).get("retrieval_wer_impact") or {}
    series, table_rows = [], []
    for idx, (pair, metrics) in enumerate(sorted(impact.items())):
        if not isinstance(metrics, dict):
            continue
        points = []
        for name, stats in metrics.items():
            if not (isinstance(stats, dict) and _num(stats.get("mean")) is not None):
                continue
            if "@" not in name:
                continue
            try:
                k = int(name.split("@", 1)[1])
            except ValueError:
                continue
            points.append((k, _num(stats["mean"])))
        if not points:
            continue
        points.sort()
        series.append({
            "type": "scatter", "mode": "markers+lines", "name": pair,
            "x": [k for k, _ in points], "y": [v for _, v in points],
            "colorkey": f"cat{(idx % 4) + 1}",
            "hovertemplate": "<b>%{y:.4f}</b> lost at k=%{x}<br>" + pair + "<extra></extra>",
        })
        table_rows += [[pair, k, v] for k, v in points]
    if not series:
        return empty_spec(chart_id, title,
                          "No retrieval-WER-impact recorded (needs a baseline branch).")
    return chart_spec(
        chart_id, title, series,
        note="Recall(baseline) − Recall(branch) at each cutoff. Higher means the branch "
             "loses more retrieval; a flat tail means a larger k stops helping.",
        layout={"xaxis": {"title": "k"}, "yaxis": {"title": "recall lost"},
                "legend": {"orientation": "h", "y": -0.25}},
        table={"caption": "Recall lost against the baseline per cutoff",
               "columns": ["Comparison", "k", "Recall lost"], "rows": table_rows},
    )


# ── Per-query distributions ──────────────────────────────────────────────────
# These read `report.traces.query_traces`, which only exists when trace_limit > 0.
# They bin SERVER-SIDE and emit bar traces: the payload stays small regardless of N,
# the bin edges are deterministic (a browser-side histogram would re-bin per render),
# and the table twin gets real numbers instead of a picture.

_TRACE_HINT = ("Per-query charts need query traces — set `trace_limit` to the dataset "
               "size in the config (0 disables traces).")


def _histogram(values: List[float], bins: int) -> "tuple[List[float], List[int], float]":
    """(bin centers, counts, width) over [min, max]; a degenerate range gets one bin."""
    lo, hi = min(values), max(values)
    if hi <= lo:
        return [lo], [len(values)], max(abs(lo), 1.0) * 0.1 or 0.1
    width = (hi - lo) / bins
    counts = [0] * bins
    for v in values:
        idx = int((v - lo) / width)
        counts[min(idx, bins - 1)] += 1          # the max value belongs in the last bin
    centers = [lo + width * (i + 0.5) for i in range(bins)]
    return centers, counts, width


def per_query_histogram(traces: Any, field: str = "per_query_wer", *, bins: int = 20,
                        chart_id: Optional[str] = None,
                        label: Optional[str] = None) -> Dict[str, Any]:
    """Distribution of a per-query score.

    The branch means hide their own shape: a 0.30 mean WER can be "every query is
    mediocre" or "most are perfect and a handful are catastrophic", and only the
    distribution distinguishes them — which is exactly the difference that matters when
    deciding whether ASR quality is a systemic or a tail problem.
    """
    label = label or field.replace("per_query_", "").upper()
    chart_id = chart_id or f"dist-{field.replace('_', '-')}"
    title = f"{label} distribution across queries"
    values = [_num(t.get(field)) for t in (traces or []) if isinstance(t, dict)]
    values = [v for v in values if v is not None]
    if not values:
        return empty_spec(chart_id, title, _TRACE_HINT)

    centers, counts, width = _histogram(values, bins)
    mean = sum(values) / len(values)
    series = [{
        "type": "bar", "x": centers, "y": counts, "width": width,
        "colorkey": "cat1",
        "hovertemplate": "<b>%{y:.0f}</b> queries near " + label + " %{x:.3f}<extra></extra>",
    }]
    return chart_spec(
        chart_id, title, series,
        note=f"n={len(values)} queries, mean {mean:.4f}. A long right tail means a few "
             f"queries carry most of the error.",
        layout={"xaxis": {"title": label.lower()}, "yaxis": {"title": "queries"},
                "bargap": 0.02,
                # the mean is the number reported elsewhere; show where it sits
                "shapes": [{"type": "line", "x0": mean, "x1": mean, "yref": "paper",
                            "y0": 0, "y1": 1, "colorkey": "neg"}]},
        table={"caption": f"Query counts per {label} bin",
               "columns": [f"{label} (bin center)", "Queries"],
               "rows": [[c, n] for c, n in zip(centers, counts)]},
    )


#: Beyond this many points an SVG scatter becomes slow and unreadable; the sample is
#: deterministic (every k-th trace) and the note says how many were shown.
_SCATTER_CAP = 2000


def wer_vs_recall_scatter(traces: Any, report: Optional[Dict[str, Any]] = None,
                          chart_id: str = "wer-vs-recall") -> Dict[str, Any]:
    """Does a worse transcript actually cost retrieval, query by query?

    The run already computes the Pearson r and prints it as one number; the scatter is
    the only honest display of a 2-D relationship — r alone cannot distinguish a real
    trend from a couple of leverage points.
    """
    title = "Transcription error vs retrieval success"
    rows = [(_num(t.get("per_query_wer")), _num(t.get("recall_at_5")))
            for t in (traces or []) if isinstance(t, dict)]
    rows = [(w, r) for w, r in rows if w is not None and r is not None]
    if not rows:
        return empty_spec(chart_id, title, _TRACE_HINT)

    total = len(rows)
    step = max(1, -(-total // _SCATTER_CAP))     # deterministic every-k-th sample
    shown = rows[::step]

    # recall@5 takes a handful of discrete values, so identical points stack invisibly.
    # Spread them with a DETERMINISTIC offset (index-derived, not random) so a re-render
    # is pixel-identical and nothing is invented that a reader could mistake for data.
    jitter = [((i % 7) - 3) * 0.012 for i in range(len(shown))]
    series = [{
        "type": "scatter", "mode": "markers", "name": "queries",
        "x": [w for w, _ in shown],
        "y": [r + j for (_, r), j in zip(shown, jitter)],
        "customdata": [r for _, r in shown],
        "colorkey": "cat1",
        "marker": {"size": 9, "opacity": 0.65},
        "hovertemplate": "WER <b>%{x:.3f}</b><br>recall@5 %{customdata:.2f}<extra></extra>",
    }]

    # Trend: mean recall per WER decile — the summary the eye should take away.
    buckets: Dict[int, List[float]] = {}
    lo = min(w for w, _ in rows)
    hi = max(w for w, _ in rows)
    if hi > lo:
        for w, r in rows:
            buckets.setdefault(min(9, int((w - lo) / ((hi - lo) / 10))), []).append(r)
        xs = [lo + ((hi - lo) / 10) * (b + 0.5) for b in sorted(buckets)]
        ys = [sum(v) / len(v) for _, v in sorted(buckets.items())]
        series.append({
            "type": "scatter", "mode": "markers+lines", "name": "mean per decile",
            "x": xs, "y": ys, "colorkey": "cat2",
            "marker": {"size": 10},
            "hovertemplate": "mean recall@5 <b>%{y:.3f}</b><extra></extra>",
        })

    r_value = _num((report or {}).get("wer_recall5_correlation"))
    note = (f"{len(shown)} of {total} queries shown. " if step > 1 else f"{total} queries. ")
    note += ("Pearson r = %.3f (computed by the run). " % r_value) if r_value is not None else ""
    note += "recall@5 is discrete, so points carry a small fixed offset to unstack them."
    return chart_spec(
        chart_id, title, series,
        note=note,
        layout={"xaxis": {"title": "per-query WER"},
                "yaxis": {"title": "recall@5"},
                "legend": {"orientation": "h", "y": -0.25}},
        table={"caption": "Mean recall@5 per WER decile",
               "columns": ["WER decile (center)", "Mean recall@5"],
               "rows": [[x, y] for x, y in zip(
                   series[-1]["x"], series[-1]["y"])] if len(series) > 1 else []},
    )


def score_margin_histogram(traces: Any, *, bins: int = 20,
                           chart_id: str = "score-margin") -> Dict[str, Any]:
    """How decisively the retriever preferred a relevant document.

    margin = score(best relevant) − score(best irrelevant). Positive means a relevant doc
    won; negative means a distractor outranked everything relevant. This separates
    "confidently right" from "barely right" — a run can hold its recall while its margins
    collapse, which is the early warning a recall number cannot give.
    """
    title = "Retrieval score margin"
    margins = []
    for t in (traces or []):
        if not isinstance(t, dict):
            continue
        relevant = t.get("relevant") or {}
        best_rel = best_irr = None
        for hit in (t.get("retrieved") or []):
            if not isinstance(hit, dict):
                continue
            score = _num(hit.get("score"))
            if score is None:
                continue
            if str(hit.get("doc_key")) in {str(k) for k in relevant}:
                best_rel = score if best_rel is None else max(best_rel, score)
            else:
                best_irr = score if best_irr is None else max(best_irr, score)
        if best_rel is not None and best_irr is not None:
            margins.append(best_rel - best_irr)
    if not margins:
        return empty_spec(
            chart_id, title,
            "Needs query traces with both a relevant and an irrelevant hit per query. "
            + _TRACE_HINT)

    centers, counts, width = _histogram(margins, bins)
    negative = sum(1 for m in margins if m < 0)
    series = [{
        "type": "bar", "x": centers, "y": counts, "width": width,
        # polarity: a negative margin is a different outcome, not a smaller one
        "colorkey": ["neg" if c < 0 else "cat1" for c in centers],
        "hovertemplate": "<b>%{y:.0f}</b> queries near margin %{x:.3f}<extra></extra>",
    }]
    return chart_spec(
        chart_id, title, series,
        note=f"{negative} of {len(margins)} queries have a NEGATIVE margin — a distractor "
             f"scored above every relevant document. Mass near zero means the ranking is "
             f"fragile even where it is correct.",
        layout={"xaxis": {"title": "score(best relevant) − score(best irrelevant)"},
                "yaxis": {"title": "queries"}, "bargap": 0.02,
                "shapes": [{"type": "line", "x0": 0, "x1": 0, "yref": "paper",
                            "y0": 0, "y1": 1, "colorkey": "zero"}]},
        table={"caption": "Query counts per score-margin bin",
               "columns": ["Margin (bin center)", "Queries"],
               "rows": [[c, n] for c, n in zip(centers, counts)]},
    )


# ── Run efficiency + cross-run ───────────────────────────────────────────────

def stage_timing(prov: Dict[str, Any], latency: Optional[Dict[str, Any]] = None,
                 chart_id: str = "stage-timing") -> Dict[str, Any]:
    """Where the wall clock went, longest stage first.

    Deliberately does NOT overlay cache hit-rate: seconds and a 0-1 rate on one plot
    would be a dual axis, whose alignment is arbitrary and invents a relationship the
    data does not contain. The cache story is its own chart below.
    """
    title = "Time per stage"
    timing = dict(prov.get("timing") or {})
    if not timing and latency:
        timing = {k: v for k, v in latency.items() if k != "total_s"}
    pairs = [(str(k), _num(v)) for k, v in timing.items()]
    pairs = [(k, v) for k, v in pairs if v is not None and v > 0]
    if not pairs:
        return empty_spec(chart_id, title, "No per-stage timings recorded for this run.")
    pairs.sort(key=lambda kv: kv[1])
    labels = [k[:-2] if k.endswith("_s") else k for k, _ in pairs]
    values = [v for _, v in pairs]
    total = sum(values)
    series = [{
        "type": "bar", "orientation": "h", "x": values, "y": labels,
        "colorkey": "cat1",
        "hovertemplate": "<b>%{x:.3f}s</b><br>%{y}<extra></extra>",
    }]
    return chart_spec(
        chart_id, title, series,
        # "summed", not "total": stages can overlap, so this exceeds the run's wall clock.
        note=f"{total:.1f}s summed across {len(values)} stages.",
        layout={"xaxis": {"title": "seconds"},
                "margin": {"l": 130, "r": 24, "t": 16, "b": 48}},
        table={"caption": "Seconds and share of total per stage",
               "columns": ["Stage", "Seconds", "Share"],
               "rows": [[k, v, f"{100.0 * v / total:.1f}%"]
                        for k, v in zip(reversed(labels), reversed(values))]},
    )


def cache_hit_rates(prov: Dict[str, Any], chart_id: str = "cache-hits") -> Dict[str, Any]:
    """Cache reuse per stage — the usual answer to "why was this run fast/slow".

    Its own panel rather than an overlay on the timing chart, so both keep an honest
    single axis.
    """
    title = "Cache reuse"
    cache = prov.get("cache") or prov.get("cache_stats") or {}
    labels, rates, table_rows = [], [], []

    def _add(name: str, stats: Dict[str, Any]) -> None:
        hits, misses = _num(stats.get("hits")) or 0, _num(stats.get("misses")) or 0
        if hits + misses <= 0:
            return
        rate = _num(stats.get("hit_rate"))
        if rate is None:
            rate = hits / (hits + misses)
        labels.append(name)
        rates.append(rate)
        table_rows.append([name, int(hits), int(misses), f"{100.0 * rate:.0f}%"])

    for stage, stats in (cache or {}).items():
        if not isinstance(stats, dict):
            continue
        # ASR nests a level deeper (features / transcriptions)
        if any(isinstance(v, dict) for v in stats.values()):
            for sub, sub_stats in stats.items():
                if isinstance(sub_stats, dict):
                    _add(f"{stage}/{sub}", sub_stats)
        else:
            _add(str(stage), stats)
    if not labels:
        return empty_spec(chart_id, title,
                          "No cache activity recorded (caching disabled for this run).")
    series = [{
        "type": "bar", "orientation": "h", "x": rates, "y": labels,
        "colorkey": "cat3",
        # Direct labels: a 0% stage draws no bar at all, so without them its row reads
        # as a rendering failure rather than as "nothing was reused here".
        "text": [f"{100.0 * r:.0f}%" for r in rates],
        "textposition": "outside",
        "hovertemplate": "<b>%{x:.0%}</b> reused<br>%{y}<extra></extra>",
    }]
    return chart_spec(
        chart_id, title, series,
        note="Share of lookups served from cache. A fast run with high reuse is not "
             "evidence about model speed.",
        layout={"xaxis": {"title": "hit rate", "range": [0, 1], "tickformat": ".0%"},
                "margin": {"l": 160, "r": 24, "t": 16, "b": 48}},
        table={"caption": "Cache hits, misses and hit rate per stage",
               "columns": ["Stage", "Hits", "Misses", "Hit rate"], "rows": table_rows},
    )


def token_budget(prov: Dict[str, Any], chart_id: str = "token-budget") -> Dict[str, Any]:
    """LLM tokens by component, against the run's budget when one is set.

    Stacked prompt/completion because they are parts of the same total the budget
    measures — and the budget itself is a rule, not another bar.
    """
    title = "LLM tokens by component"
    cost = prov.get("cost") or {}
    by_component = cost.get("by_component") or {}
    rows = [(str(k), v) for k, v in by_component.items() if isinstance(v, dict)]
    if not rows:
        return empty_spec(chart_id, title, "No LLM calls in this run.")
    labels = [k for k, _ in rows]
    prompt = [_num(v.get("prompt_tokens")) or 0 for _, v in rows]
    completion = [_num(v.get("completion_tokens")) or 0 for _, v in rows]
    series = [
        {"type": "bar", "orientation": "h", "name": "prompt", "x": prompt, "y": labels,
         "colorkey": "cat1",
         "hovertemplate": "<b>%{x:.0f}</b> prompt tokens<br>%{y}<extra></extra>"},
        {"type": "bar", "orientation": "h", "name": "completion", "x": completion,
         "y": labels, "colorkey": "cat2",
         "hovertemplate": "<b>%{x:.0f}</b> completion tokens<br>%{y}<extra></extra>"},
    ]
    layout = {"barmode": "stack", "xaxis": {"title": "tokens"},
              "margin": {"l": 150, "r": 24, "t": 16, "b": 48},
              "legend": {"orientation": "h", "y": -0.3}}
    budget = _num(cost.get("budget_tokens"))
    total = _num((cost.get("totals") or {}).get("total_tokens")) or sum(prompt) + sum(completion)
    note = f"{total:.0f} tokens total."
    if budget:
        layout["shapes"] = [{"type": "line", "x0": budget, "x1": budget, "yref": "paper",
                             "y0": 0, "y1": 1, "colorkey": "neg"}]
        note += f" Budget {budget:.0f} ({100.0 * total / budget:.0f}% used)."
    return chart_spec(
        chart_id, title, series, note=note, layout=layout,
        table={"caption": "Prompt and completion tokens per component",
               "columns": ["Component", "Prompt", "Completion", "Calls"],
               "rows": [[k, _num(v.get("prompt_tokens")) or 0,
                         _num(v.get("completion_tokens")) or 0,
                         _num(v.get("calls")) or 0] for k, v in rows]},
    )


def leaderboard_scatter(rows: Any, metric: str, xfield: str = "duration_seconds",
                        chart_id: Optional[str] = None) -> Dict[str, Any]:
    """A metric against cost (wall clock) or against time (run order).

    Both fields already ride on every leaderboard row and nothing plotted them. Quality
    against duration is the "best at what price" view the Pareto page cannot reach when
    duration is not one of its objectives; quality against created_at answers "is this
    sweep actually improving".
    """
    over_time = xfield == "created_at"
    chart_id = chart_id or ("metric-over-time" if over_time else "metric-vs-duration")
    title = (f"{metric} over time" if over_time else f"{metric} vs run duration")
    points = []
    for r in (rows or []):
        if not isinstance(r, dict):
            continue
        y = _num(r.get("metric_value"))
        x = r.get(xfield) if over_time else _num(r.get(xfield))
        if y is None or x is None:
            continue
        points.append((x, y, str(r.get("experiment_name") or r.get("run_id") or "")))
    xlabel = "a finish time" if over_time else "a run duration"
    if not points:
        return empty_spec(
            chart_id, title, f"No runs carry both {metric} and {xlabel} yet.")
    if over_time and len(points) < 2:
        # One point on a time axis gives Plotly a zero-width range, which it pads out to
        # milliseconds and labels 22:24:43.999 … 22:24:44.001 — a broken-looking chart
        # standing in for "there is no trend yet".
        return empty_spec(chart_id, title,
                          "A trend needs at least two runs; this filter matches one.")
    points.sort(key=lambda p: p[0])
    series = [{
        "type": "scatter",
        "mode": "markers+lines" if over_time else "markers",
        "x": [p[0] for p in points], "y": [p[1] for p in points],
        "text": [p[2] for p in points], "colorkey": "cat1",
        "marker": {"size": 11},
        "hovertemplate": "<b>%{y:.4f}</b> " + metric + "<br>%{x}<br>%{text}<extra></extra>",
    }]
    return chart_spec(
        chart_id, title, series,
        note=("Runs in chronological order." if over_time else
              "Up and to the left is better: the same quality for less wall clock."),
        layout={"xaxis": {"title": "run finished" if over_time else "seconds"},
                "yaxis": {"title": metric}},
        table={"caption": f"{metric} per run against {xfield}",
               "columns": ["Run", xfield, metric],
               "rows": [[p[2], p[0], p[1]] for p in points]},
    )


def compare_relative_deltas(metric_rows: Any, *, top_n: int = 14,
                            chart_id: str = "compare-deltas") -> Dict[str, Any]:
    """Which metrics moved between two runs, as a relative change.

    Relative (%) rather than raw, because a compare table mixes 0-1 rates with
    millisecond latencies — putting their raw deltas on one axis would let a latency
    swing of 40ms dwarf a recall change of 0.4 and read as the bigger effect.

    Deliberately carries NO error bars: these are two independent runs, not a paired
    comparison, so there is no confidence interval and no significance test behind the
    numbers. The note says so, because a bar chart of differences invites exactly that
    misreading.
    """
    title = "Relative change, B vs A"
    rows = []
    for r in (metric_rows or []):
        if not isinstance(r, dict):
            continue
        a, b = _num(r.get("a")), _num(r.get("b"))
        name = str(r.get("name") or "")
        if a is None or b is None or not name:
            continue
        if a == 0:
            continue          # a relative change against zero is undefined, not infinite
        rows.append((name, 100.0 * (b - a) / abs(a), a, b))
    if not rows:
        return empty_spec(chart_id, title,
                          "These two runs share no metric with a non-zero baseline value.")

    rows.sort(key=lambda t: abs(t[1]), reverse=True)
    shown = list(reversed(rows[:top_n]))      # biggest movers nearest the top after plotting
    series = [{
        "type": "bar", "orientation": "h",
        "x": [pct for _, pct, _, _ in shown],
        "y": [name for name, _, _, _ in shown],
        # polarity: up vs down, so the diverging pair rather than one identity hue
        "colorkey": ["pos" if pct >= 0 else "neg" for _, pct, _, _ in shown],
        "hovertemplate": "<b>%{x:+.1f}%</b><br>%{y}<extra></extra>",
    }]
    note = ("Unpaired: two independent runs, so there is no confidence interval and no "
            "significance test here — use a branched run for that. ")
    if len(rows) > top_n:
        note += f"Showing the {top_n} largest movers of {len(rows)}."
    return chart_spec(
        chart_id, title, series,
        note=note,
        layout={"xaxis": {"title": "change vs A (%)", "ticksuffix": "%"},
                "margin": {"l": 170, "r": 32, "t": 16, "b": 48},
                "shapes": [{"type": "line", "x0": 0, "x1": 0, "yref": "paper",
                            "y0": 0, "y1": 1, "colorkey": "zero"}]},
        table={"caption": "Metric values for both runs and the relative change",
               "columns": ["Metric", "A", "B", "Change"],
               "rows": [[n, a, b, f"{pct:+.1f}%"] for n, pct, a, b in rows]},
    )


def metric_chips(metrics: Any, branches: Any = None) -> Dict[str, Any]:
    """The run's scalar metrics, each shown once.

    A run stores the same number up to three times: under a display name (``MRR``), under
    its branch (``asr_text_retrieval/mrr``), and again as that branch's mean in the report.
    Rendered straight, one single-branch run puts 39 chips on screen carrying 15 distinct
    facts, and the headline metric gets exactly as much weight as the 25th repeat.

    Deduped on (metric name without its branch prefix, the value as displayed). Two
    branches that genuinely disagree therefore keep both chips — only the repeats go.
    """
    src = metrics if isinstance(metrics, dict) else {}
    seen: set = set()
    chips: List[Dict[str, Any]] = []

    def _text(value: Any) -> Optional[str]:
        # Flags are not measurements: `phased` is True, not 1.0000.
        if isinstance(value, bool):
            return "yes" if value else "no"
        num = _num(value)
        return None if num is None else f"{num:.4f}"

    def _consider(name: str) -> None:
        if name.endswith("_ci"):
            return                      # rendered as a range on its own metric's chip
        text = _text(src.get(name))
        if text is None:
            return
        key = (name.rsplit("/", 1)[-1].lower(), text)
        if key in seen:
            return
        seen.add(key)
        ci = src.get(f"{name}_ci")
        chips.append({
            "label": name, "text": text,
            "ci": [_num(ci[0]), _num(ci[1])]
            if isinstance(ci, (list, tuple)) and len(ci) == 2 else None,
        })

    # Unprefixed names first, so the chip that survives is the readable one.
    for name in [k for k in src if "/" not in str(k)]:
        _consider(str(name))
    for name in [k for k in src if "/" in str(k)]:
        _consider(str(name))

    # Does the per-branch block still say anything the chips did not?
    extra = False
    for bmetrics in (branches or {}).values():
        if not isinstance(bmetrics, dict):
            continue
        for mname, stats in bmetrics.items():
            if not isinstance(stats, dict):
                continue
            text = _text(stats.get("mean"))
            if text is not None and (str(mname).rsplit("/", 1)[-1].lower(), text) not in seen:
                extra = True
    return {"chips": chips, "branch_values": extra}
