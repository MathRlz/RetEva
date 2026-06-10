"""Aggregation: reduce per-item scores → report, with cross-branch deltas (architecture A5/A9).

Metric nodes emit per-item ``item_scores`` (an ``ItemSet``); the aggregate node owns ALL
reduction — means (and optional bootstrap CIs) per branch, plus **paired** cross-branch deltas
(e.g. ``ΔRecall@k(q) = Recall_corr(q) − Recall_asr(q)`` averaged over shared query ids). Pairing
is by id, which is why every branch must carry the same ``query_id`` (the ``ItemSet``).

See ``evaluator-architecture.md`` §8.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from .item_set import ItemSet

MIN_SAMPLES_FOR_CI = 20
_BOOTSTRAP_ALPHA = 0.05
_BOOTSTRAP_ITERS = 1000


def reduce_scores(item_scores: ItemSet, *, with_ci: bool = False) -> Dict[str, Any]:
    """Reduce a per-item ``item_scores`` set to ``{mean, n}`` (+ bootstrap ``ci`` when asked
    and enough samples)."""
    values = [float(v) for _, v in item_scores]
    n = len(values)
    out: Dict[str, Any] = {"mean": (sum(values) / n) if n else 0.0, "n": n}
    if with_ci and n >= MIN_SAMPLES_FOR_CI:
        try:
            from ..analysis.significance import bootstrap_confidence_interval

            out["ci"] = bootstrap_confidence_interval(
                values, alpha=_BOOTSTRAP_ALPHA, n_bootstrap=_BOOTSTRAP_ITERS
            )
        except Exception:
            pass
    return out


def paired_delta(branch: ItemSet, baseline: ItemSet) -> Dict[str, Any]:
    """Mean paired difference ``branch − baseline`` over the shared ids (id-aligned).

    The delta is the only sound paired comparison, so it is computed over the *intersection*
    of ids — but the honest denominator is reported alongside (S1): how many items each side
    had and how many were dropped because they were present in only one branch (e.g. an item
    that failed/was dropped in one branch). A silent intersection would hide a shrinking,
    asymmetric sample."""
    branch_ids, base_ids = set(branch.ids), set(baseline.ids)
    ids, b_vals, base_vals = branch.align(baseline)
    diffs = [float(b) - float(a) for b, a in zip(b_vals, base_vals)]
    n = len(diffs)
    return {
        "mean_delta": (sum(diffs) / n) if n else 0.0,
        "n_paired": n,
        "n_branch": len(branch),
        "n_baseline": len(baseline),
        "n_only_branch": len(branch_ids - base_ids),
        "n_only_baseline": len(base_ids - branch_ids),
        "denominator_policy": "intersection",
        "_diffs": diffs,  # consumed by S3 (CI + paired test); stripped before serialize
    }


def flatten_report(report: Mapping[str, Any]) -> Dict[str, float]:
    """Flatten a report's per-branch means + deltas + impact to top-level numeric keys
    (``"<branch>/<metric>"``, ``"delta_<metric>(<b_vs_base>)"``, ``"impact_<metric>(...)"``)
    so the leaderboard auto-discovers + sorts them (`available_metrics`/`query_leaderboard`).
    ASCII keys (no Δ) so they're safe in URL sort params (M4)."""
    out: Dict[str, float] = {}
    for branch, metrics in (report.get("branches") or {}).items():
        for name, stats in metrics.items():
            mean = stats.get("mean") if isinstance(stats, dict) else None
            if isinstance(mean, (int, float)) and not isinstance(mean, bool):
                out[f"{branch}/{name}"] = float(mean)
    for pair, metrics in (report.get("deltas") or {}).items():
        for name, stats in metrics.items():
            d = stats.get("mean_delta") if isinstance(stats, dict) else None
            if isinstance(d, (int, float)) and not isinstance(d, bool):
                out[f"delta_{name}({pair})"] = float(d)
    for pair, metrics in (report.get("retrieval_wer_impact") or {}).items():
        for name, stats in metrics.items():
            m = stats.get("mean") if isinstance(stats, dict) else None
            if isinstance(m, (int, float)) and not isinstance(m, bool):
                out[f"impact_{name}({pair})"] = float(m)
    return out


def _enrich_deltas(deltas: Dict[str, Dict[str, Any]]) -> None:
    """In-place: add a paired-bootstrap CI + paired test p-value + paired Cohen's d to each
    delta (S3), then a Benjamini-Hochberg FDR q-value + `significant` flag across the whole
    family of delta tests (S4). Strips the internal `_diffs`. Deterministic CI (seeded).
    """
    import numpy as np

    from ..analysis.significance import (
        benjamini_hochberg,
        bootstrap_confidence_interval,
        wilcoxon_test,
    )

    pkeys: list = []  # (pair, metric) for each computed p-value, in order
    pvals: list = []
    for pair, dmetrics in deltas.items():
        for name, d in dmetrics.items():
            diffs = d.pop("_diffs", [])
            n = d.get("n_paired", 0)
            if n < 2 or all(x == 0.0 for x in diffs):
                continue
            sd = float(np.std(diffs, ddof=1))
            d["cohens_d"] = (d["mean_delta"] / sd) if sd > 0 else 0.0
            try:
                d["ci"] = list(bootstrap_confidence_interval(diffs, random_state=0))
            except Exception:
                pass
            try:
                _, p = wilcoxon_test(diffs, [0.0] * n)
                d["p_value"] = float(p)
                pkeys.append((pair, name))
                pvals.append(float(p))
            except Exception:
                pass
    if pvals:
        for (pair, name), q in zip(pkeys, benjamini_hochberg(pvals)):
            deltas[pair][name]["p_value_fdr"] = q
            deltas[pair][name]["significant"] = bool(q < 0.05)


def build_report(
    per_branch_metrics: Mapping[str, Mapping[str, ItemSet]],
    *,
    baseline: Optional[str] = None,
    with_ci: bool = False,
    provenance: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the run ``report`` from ``{branch -> {metric -> item_scores}}``.

    Produces per-branch reduced metrics and, when a ``baseline`` branch is given, paired
    deltas of every other branch's metrics against it (only for metrics both branches share).
    """
    branches: Dict[str, Dict[str, Any]] = {}
    for branch_id, metrics in per_branch_metrics.items():
        branches[branch_id] = {
            name: reduce_scores(scores, with_ci=with_ci)
            for name, scores in metrics.items()
        }

    report: Dict[str, Any] = {"branches": branches}

    if baseline is not None and baseline in per_branch_metrics:
        base_metrics = per_branch_metrics[baseline]
        deltas: Dict[str, Dict[str, Any]] = {}
        for branch_id, metrics in per_branch_metrics.items():
            if branch_id == baseline:
                continue
            shared = {
                name: paired_delta(scores, base_metrics[name])
                for name, scores in metrics.items()
                if name in base_metrics
            }
            if shared:
                deltas[f"{branch_id}_vs_{baseline}"] = shared
        if deltas:
            _enrich_deltas(deltas)  # S3 (CI + paired test) + S4 (FDR + effect size)
            report["deltas"] = deltas

    if provenance is not None:
        report["provenance"] = dict(provenance)
    return report
