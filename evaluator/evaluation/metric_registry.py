"""Metric-spec registry + per-item computation (architecture A4).

A metric is declared once with the artifacts it needs — a *scored* artifact and an optional
*ground-truth* artifact — and a per-item scoring function. The builder/runtime can then
**auto-inject** a metric wherever its declared inputs are present (default = the dataset's
mode set; opt-in ``collect_all`` = every metric whose inputs are satisfiable).

Computation is keyed: :func:`compute_metric` ``align``s the scored and GT ``ItemSet``s by id
(gaps tolerated) and emits a per-item ``item_scores`` ``ItemSet``. Reduction to scalars and
cross-branch deltas is the aggregate node's job (see ``aggregate.py``), per the strict
per-item / aggregate split.

Underlying scoring reuses ``metrics/ir.py`` + ``metrics/stt.py``. See
``evaluator-architecture.md`` §7.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from ..metrics.clinical import critical_entity_error_rate
from ..metrics.ir import (
    average_precision,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from ..metrics.stt import character_error_rate, word_error_rate
from .item_set import ItemSet

# A per-item scoring function: ``fn(scored_value, gt_value) -> float`` (gt None when absent).
ScoreFn = Callable[[Any, Any], float]


@dataclass(frozen=True)
class MetricSpec:
    """A registered metric: name, the artifacts it consumes, and its per-item scorer."""

    name: str
    scored: str  # artifact name carrying the thing being scored
    fn: ScoreFn
    gt: Optional[str] = None  # ground-truth artifact name (None = reference-free)
    higher_is_better: bool = True

    @property
    def inputs(self) -> Sequence[str]:
        return (self.scored, self.gt) if self.gt else (self.scored,)


_METRIC_REGISTRY: Dict[str, MetricSpec] = {}


def register_metric(
    name: str,
    *,
    scored: str,
    fn: ScoreFn,
    gt: Optional[str] = None,
    higher_is_better: bool = True,
) -> MetricSpec:
    """Register a metric spec (idempotent for an identical re-registration)."""
    spec = MetricSpec(
        name=name, scored=scored, fn=fn, gt=gt, higher_is_better=higher_is_better
    )
    existing = _METRIC_REGISTRY.get(name)
    if existing is not None and existing != spec:
        raise ValueError(f"metric '{name}' already registered as {existing}")
    _METRIC_REGISTRY[name] = spec
    return spec


def get_metric(name: str) -> MetricSpec:
    if name not in _METRIC_REGISTRY:
        known = ", ".join(sorted(_METRIC_REGISTRY))
        raise KeyError(f"unknown metric '{name}'. Registered: {known}")
    return _METRIC_REGISTRY[name]


def list_metrics() -> List[MetricSpec]:
    return [_METRIC_REGISTRY[name] for name in sorted(_METRIC_REGISTRY)]


def applicable_metrics(
    available: Sequence[str],
    *,
    default: Optional[Sequence[str]] = None,
    collect_all: bool = False,
) -> List[MetricSpec]:
    """The metric specs that can run given the ``available`` artifact names.

    ``collect_all`` → every metric whose inputs are all present. Otherwise only those in
    ``default`` (the dataset's ``evaluation_mode`` set) whose inputs are present.
    """
    have = set(available)
    out: List[MetricSpec] = []
    for spec in list_metrics():
        if not set(spec.inputs) <= have:
            continue
        if collect_all or default is None or spec.name in set(default):
            out.append(spec)
    return out


def compute_metrics(
    artifacts: Mapping[str, ItemSet],
    *,
    default: Optional[Sequence[str]] = None,
    collect_all: bool = False,
) -> Dict[str, ItemSet]:
    """Run every applicable metric over the available keyed ``artifacts``.

    Returns ``{metric_name: item_scores}``. The applicable set is chosen by
    :func:`applicable_metrics` (default mode set, or all satisfiable with ``collect_all``).
    This is what a per-branch metrics node calls; the ``aggregate`` node then reduces the
    returned ``item_scores`` to scalars + cross-branch deltas.
    """
    specs = applicable_metrics(
        list(artifacts.keys()), default=default, collect_all=collect_all
    )
    return {spec.name: compute_metric(spec, artifacts) for spec in specs}


def compute_metric(spec: MetricSpec, artifacts: Mapping[str, ItemSet]) -> ItemSet:
    """Compute a metric's per-item ``item_scores`` from the resolved input ``ItemSet``s.

    Aligns the scored and (optional) GT sets by id — so a sparse/failed item simply does
    not contribute — and applies the per-item ``fn``.
    """
    scored = artifacts[spec.scored]
    if spec.gt is None:
        return scored.map_values(lambda v: float(spec.fn(v, None)))
    ids, scored_vals, gt_vals = scored.align(artifacts[spec.gt])
    return ItemSet(ids, [float(spec.fn(s, g)) for s, g in zip(scored_vals, gt_vals)])


# ── built-in metrics ──────────────────────────────────────────────────
# ASR: scored = the ASR hypothesis (query_text, now immutable since correction/optimization
# emit distinct names), gt = reference_text. So `wer`/`cer` always measure ASR quality —
# the old raw_wer/raw_cer (which scored a separate raw_query_text snapshot) are subsumed.
# A "corrected WER" is now expressible by pointing a second metric at corrected_query_text.
register_metric(
    "wer",
    scored="query_text",
    gt="reference_text",
    fn=lambda hyp, ref: word_error_rate(ref, hyp),
    higher_is_better=False,
)
register_metric(
    "cer",
    scored="query_text",
    gt="reference_text",
    fn=lambda hyp, ref: character_error_rate(ref, hyp),
    higher_is_better=False,
)
# CEER — critical-entity (drug/dose/unit) error rate; a low WER can hide a dangerous error.
# Supersedes the legacy term-weighted WER (`TW_WER`, L2): TW_WER crudely up-weighted a broad
# medical vocabulary, whereas CEER targets the actually-dangerous drug/dose/unit entities.
register_metric(
    "ceer",
    scored="query_text",
    gt="reference_text",
    fn=lambda hyp, ref: critical_entity_error_rate(ref, hyp),
    higher_is_better=False,
)


# IR: scored = retrieved doc-id keys (ranked list), gt = relevant_docs (doc_id → grade).
def _at_k(metric_fn: Callable[[Any, Any, int], float], k: int) -> ScoreFn:
    """Bind a rank cutoff into a (scored, gt) -> float scorer."""

    def fn(scored: Any, gt: Any) -> float:
        return float(metric_fn(scored, gt, k))

    return fn


for _k in (1, 5, 10):
    register_metric(
        f"recall@{_k}",
        scored="retrieved",
        gt="relevant_docs",
        fn=_at_k(recall_at_k, _k),
    )
    register_metric(
        f"precision@{_k}",
        scored="retrieved",
        gt="relevant_docs",
        fn=_at_k(precision_at_k, _k),
    )
    register_metric(
        f"ndcg@{_k}", scored="retrieved", gt="relevant_docs", fn=_at_k(ndcg_at_k, _k)
    )
register_metric(
    "mrr",
    scored="retrieved",
    gt="relevant_docs",
    fn=lambda ret, rel: float(reciprocal_rank(ret, rel)),
)
# Mean average precision (per-query AP; the run mean is MAP). Ported into the registry so the
# report is a true superset of the legacy IR scalars ahead of the greenfield cutout (G5).
register_metric(
    "map",
    scored="retrieved",
    gt="relevant_docs",
    fn=lambda ret, rel: float(average_precision(ret, rel)),
)

# --- LLM judge metrics (reference-free) ---------------------------------------------------------
# The judge node publishes per-query numeric ItemSets; each metric scores its own artifact, so a
# metric auto-injects only when that artifact is present (i.e. only the *configured* aspects fire).
# fn is the identity — the score is already in [0, 1] (pass is 1.0/0.0 → its mean is the pass rate).
from ..config.judge import VALID_JUDGE_ASPECTS  # noqa: E402

register_metric("judge_overall", scored="judge_scores", gt=None, fn=lambda v, _g: float(v))
register_metric("judge_pass_rate", scored="judge_pass", gt=None, fn=lambda v, _g: float(v))
for _aspect in sorted(VALID_JUDGE_ASPECTS):
    register_metric(
        f"judge_{_aspect}", scored=f"judge_aspect_{_aspect}", gt=None,
        fn=lambda v, _g: float(v),
    )
