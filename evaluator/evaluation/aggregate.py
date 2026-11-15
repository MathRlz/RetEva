"""Aggregation: reduce per-item scores to a report with cross-branch deltas. Metric nodes emit
per-item ``item_scores``; the aggregate node owns ALL reduction — means (+ optional bootstrap
CIs) per branch, plus **paired** deltas by id, which is why every branch must share ``query_id``."""

from __future__ import annotations

import numbers
from typing import Any, Dict, Mapping, Optional

from ..constants import ROLLUP_REDUCERS
from ..logging_config import get_logger
from .item_set import LINEAGE_SEPARATOR, ItemSet, lineage_parent
from ..analysis.significance import (
    benjamini_hochberg,
    bootstrap_confidence_interval,
    wilcoxon_test,
)

logger = get_logger(__name__)

MIN_SAMPLES_FOR_CI = 20
_BOOTSTRAP_ALPHA = 0.05
_BOOTSTRAP_ITERS = 1000
#: Fraction of the paired sample above which one-sided exclusions mark a delta drop-biased:
#: items land outside the intersection because a branch dropped them, and hard items fail
#: more (not-missing-at-random) — the honest denominators alone don't say when that starts
#: to matter.
DROP_BIAS_THRESHOLD = 0.05

_REDUCE_FNS = {
    "mean": lambda xs: sum(xs) / len(xs),
    "min": min,
    "max": max,
}
#: Keyed by the config-level name tuple, so an accepted name without a reducer here is an
#: import-time KeyError rather than a run-time one.
_ROLLUP_REDUCERS = {name: _REDUCE_FNS[name] for name in ROLLUP_REDUCERS}


def rollup_variants(item_scores: ItemSet, how: str = "mean") -> ItemSet:
    """Reduce lineage variants (``q42·aug0/1``) to one parent-level score (``q42``). Variants
    of one parent are correlated, so treating them as independent items would break the
    paired cross-branch join and make bootstrap/Wilcoxon resample clusters as if iid (too
    narrow CIs) — rolling up first makes every downstream reduction operate on the
    cluster-safe unit. ``how`` (mean/min/max) picks the reducer; worst-case is min or max
    depending on the metric's direction. Identity when no variant ids are present."""
    ids = item_scores.ids
    if not any(LINEAGE_SEPARATOR in i for i in ids):
        return item_scores
    reduce_fn = _ROLLUP_REDUCERS[how]
    groups: Dict[str, list] = {}
    order = []
    for item_id, value in item_scores:
        parent = lineage_parent(item_id)
        if parent not in groups:
            groups[parent] = []
            order.append(parent)
        groups[parent].append(float(value))
    return ItemSet(order, [reduce_fn(groups[p]) for p in order])


def reduce_scores(item_scores: ItemSet, *, with_ci: bool = False) -> Dict[str, Any]:
    """Reduce a per-item ``item_scores`` set to ``{mean, n}`` (+ bootstrap ``ci`` when asked
    and enough samples)."""
    values = [float(v) for _, v in item_scores]
    n = len(values)
    # n == 0 → mean None, not 0.0: no data must not read as a real (perfect/zero) score
    out: Dict[str, Any] = {"mean": (sum(values) / n) if n else None, "n": n}
    if with_ci and n >= MIN_SAMPLES_FOR_CI:
        try:
            out["ci"] = bootstrap_confidence_interval(
                values, alpha=_BOOTSTRAP_ALPHA, n_bootstrap=_BOOTSTRAP_ITERS,
                random_state=0,  # deterministic, like the delta CI below
            )
        except Exception as exc:
            logger.warning("bootstrap CI failed (n=%d): %s", n, exc)
            out["ci_error"] = str(exc)
    return out


def paired_delta(branch: ItemSet, baseline: ItemSet) -> Dict[str, Any]:
    """Mean paired difference ``branch − baseline`` over the shared ids (id-aligned). The
    delta is the only sound paired comparison, computed over the *intersection* of ids —
    but the honest denominator is reported alongside (how many items each side had, how
    many were dropped because present in only one branch), so a silent intersection never
    hides a shrinking, asymmetric sample."""
    branch_ids, base_ids = set(branch.ids), set(baseline.ids)
    ids, b_vals, base_vals = branch.align(baseline)
    diffs = [float(b) - float(a) for b, a in zip(b_vals, base_vals)]
    n = len(diffs)
    out = {
        "mean_delta": (sum(diffs) / n) if n else 0.0,
        "n_paired": n,
        "n_branch": len(branch),
        "n_baseline": len(baseline),
        "n_only_branch": len(branch_ids - base_ids),
        "n_only_baseline": len(base_ids - branch_ids),
        "denominator_policy": "intersection",
        "_diffs": diffs,  # consumed downstream (CI + paired test); stripped before serialize
    }
    # One-sided exclusions beyond the threshold mark the delta drop-biased — the intersection
    # then measures the easy survivors, not the branch (key absent otherwise, so a clean run's
    # report is unchanged).
    excluded = out["n_only_branch"] + out["n_only_baseline"]
    if excluded > DROP_BIAS_THRESHOLD * max(n, 1):
        out["drop_biased"] = True
    return out


def flatten_report(report: Mapping[str, Any]) -> Dict[str, float]:
    """Flatten a report's per-branch means + deltas + impact to top-level numeric keys
    (``"<branch>/<metric>"``, ``"delta_<metric>(<b_vs_base>)"``, ``"impact_<metric>(...)"``)
    so the leaderboard auto-discovers + sorts them (`available_metrics`/`query_leaderboard`).
    ASCII keys (no Δ) so they're safe in URL sort params."""
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
    """In-place: adds a paired-bootstrap CI + paired test p-value + paired Cohen's d to each
    delta, then a Benjamini-Hochberg FDR q-value + `significant` flag across the whole
    family of delta tests. Strips the internal `_diffs`; CI is deterministic (seeded)."""
    import numpy as np

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
            except Exception as exc:
                logger.warning("delta CI failed for %s/%s: %s", pair, name, exc)
                d["ci_error"] = str(exc)
            try:
                _, p = wilcoxon_test(diffs, [0.0] * n)
                d["p_value"] = float(p)
                pkeys.append((pair, name))
                pvals.append(float(p))
            except Exception as exc:
                logger.warning("paired test failed for %s/%s: %s", pair, name, exc)
                d["p_value_error"] = str(exc)
    n_failed = sum(
        1 for dmetrics in deltas.values() for d in dmetrics.values() if "p_value_error" in d
    )
    if n_failed:
        # failed tests silently shrink the BH family (FDR denominators) — make it auditable
        logger.warning(
            "BH family: %d delta test(s) excluded (test failure), %d in family",
            n_failed, len(pvals),
        )
    if pvals:
        for (pair, name), q in zip(pkeys, benjamini_hochberg(pvals)):
            deltas[pair][name]["p_value_fdr"] = q
            deltas[pair][name]["significant"] = bool(q < 0.05)
    flagged = [
        f"{pair}/{name}"
        for pair, dmetrics in deltas.items()
        for name, d in dmetrics.items()
        if d.get("drop_biased")
    ]
    if flagged:
        logger.warning(
            "drop-biased paired deltas (one-sided exclusions > %.0f%% of n_paired — the "
            "intersection measures survivors, not the branch): %s",
            DROP_BIAS_THRESHOLD * 100,
            "; ".join(flagged),
        )


def build_report(
    per_branch_metrics: Mapping[str, Mapping[str, ItemSet]],
    *,
    baseline: Optional[str] = None,
    with_ci: bool = False,
    provenance: Optional[Mapping[str, Any]] = None,
    variant_rollup: str = "mean",
) -> Dict[str, Any]:
    """Build the run ``report`` from ``{branch -> {metric -> item_scores}}``. Produces
    per-branch reduced metrics and, when a ``baseline`` branch is given, paired deltas of
    every other branch's metrics against it (only for metrics both branches share). Lineage
    variants are rolled up to parent level first, so pairing and every CI/test below
    operates on the cluster-safe unit."""
    per_branch_metrics = {
        branch_id: {
            name: rollup_variants(scores, variant_rollup)
            for name, scores in metrics.items()
        }
        for branch_id, metrics in per_branch_metrics.items()
    }
    branches: Dict[str, Dict[str, Any]] = {}
    for branch_id, metrics in per_branch_metrics.items():
        branches[branch_id] = {
            name: reduce_scores(scores, with_ci=with_ci)
            for name, scores in metrics.items()
        }

    report: Dict[str, Any] = {"branches": branches}
    # Per-query scores (post-rollup, keyed by query id) so cross-run paired tests
    # (scripts/paired_stats.py) can join two result files offline.
    report["item_scores"] = {
        branch_id: {
            name: {i: float(v) for i, v in scores if isinstance(v, numbers.Real)}
            for name, scores in metrics.items()
        }
        for branch_id, metrics in per_branch_metrics.items()
    }

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
            _enrich_deltas(deltas)  # CI + paired test, FDR q-values, effect size
            report["deltas"] = deltas

    if provenance is not None:
        report["provenance"] = dict(provenance)
    return report
