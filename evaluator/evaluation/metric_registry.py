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
from functools import lru_cache
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from ..metrics.clinical import critical_entity_error_rate
from ..metrics.ir import (
    IR_CUTOFFS,
    average_precision,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from ..metrics.stt import character_error_rate, word_error_rate
from .item_set import ItemSet, lineage_parent

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
    opt_in: bool = False  # computed only when the caller passes include_opt_in (C7)

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
    opt_in: bool = False,
) -> MetricSpec:
    """Register a metric spec (idempotent for an identical re-registration)."""
    spec = MetricSpec(
        name=name, scored=scored, fn=fn, gt=gt,
        higher_is_better=higher_is_better, opt_in=opt_in,
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
    available: Sequence[str], *, include_opt_in: bool = False
) -> List[MetricSpec]:
    """The metric specs whose inputs are all present in ``available``. Specs registered
    ``opt_in`` are excluded unless ``include_opt_in`` (so they are never even computed on
    a run that did not enable them)."""
    have = set(available)
    return [
        spec for spec in list_metrics()
        if set(spec.inputs) <= have and (include_opt_in or not spec.opt_in)
    ]


def compute_metrics(
    artifacts: Mapping[str, ItemSet],
    *,
    include_opt_in: bool = False,
    only: Optional[Sequence[str]] = None,
) -> Dict[str, ItemSet]:
    """Run every applicable metric over the available keyed ``artifacts``.

    Returns ``{metric_name: item_scores}`` (the metrics whose inputs are all present). This is
    what a per-branch metrics node calls; the ``aggregate`` node then reduces the returned
    ``item_scores`` to scalars + cross-branch deltas.

    ``only`` (the config's ``metrics:`` allowlist, B1) computes exactly those registered
    names — explicit naming bypasses ``opt_in``; a name whose inputs the artifacts don't
    carry is skipped (unknown names raise via :func:`get_metric`, but config validation
    rejects them earlier). ``None`` keeps collect-all.
    """
    if only is not None:
        have = set(artifacts.keys())
        seen: Dict[str, MetricSpec] = {}
        for name in only:
            spec = get_metric(name)
            if name not in seen and set(spec.inputs) <= have:
                seen[name] = spec
        specs: List[MetricSpec] = list(seen.values())
    else:
        specs = applicable_metrics(list(artifacts.keys()), include_opt_in=include_opt_in)
    return {spec.name: compute_metric(spec, artifacts) for spec in specs}


def compute_metric(spec: MetricSpec, artifacts: Mapping[str, ItemSet]) -> ItemSet:
    """Compute a metric's per-item ``item_scores`` from the resolved input ``ItemSet``s.

    Aligns the scored and (optional) GT sets by id — so a sparse/failed item simply does
    not contribute — and applies the per-item ``fn``. A fan-out variant (``q42·aug0``)
    whose own id is absent from the GT set scores against its lineage parent's GT (``q42``)
    — GT artifacts are published at parent granularity by ``dataset_source``.
    """
    scored = artifacts[spec.scored]
    if spec.gt is None:
        return scored.map_values(lambda v: float(spec.fn(v, None)))
    ids, scored_vals, gt_vals = scored.align(artifacts[spec.gt], other_key=lineage_parent)
    return ItemSet(ids, [float(spec.fn(s, g)) for s, g in zip(scored_vals, gt_vals)])


# ── built-in metrics ──────────────────────────────────────────────────
# ASR: scored = the ASR hypothesis (query_text, now immutable since correction/optimization
# emit distinct names), gt = reference_transcription — the SPOKEN transcription the report
# paths publish into this slot (formerly misnamed "reference_text", which is a different,
# question-text bus artifact). `wer`/`cer` always measure ASR quality — the old
# raw_wer/raw_cer (which scored a separate raw_query_text snapshot) are subsumed.
# A "corrected WER" is now expressible by pointing a second metric at corrected_query_text.
register_metric(
    "wer",
    scored="query_text",
    gt="reference_transcription",
    fn=lambda hyp, ref: word_error_rate(ref, hyp),
    higher_is_better=False,
)
register_metric(
    "cer",
    scored="query_text",
    gt="reference_transcription",
    fn=lambda hyp, ref: character_error_rate(ref, hyp),
    higher_is_better=False,
)
# CEER — critical-entity (drug/dose/unit) error rate; a low WER can hide a dangerous error.
# Supersedes the legacy term-weighted WER (`TW_WER`, L2): TW_WER crudely up-weighted a broad
# medical vocabulary, whereas CEER targets the actually-dangerous drug/dose/unit entities.
register_metric(
    "ceer",
    scored="query_text",
    gt="reference_transcription",
    fn=lambda hyp, ref: critical_entity_error_rate(ref, hyp),
    higher_is_better=False,
)


# ── C7 opt-in metrics (query_correction.corrected_metrics, default off) ──
# corrected_* score the CORRECTED text against the reference — the default wer/cer/ceer
# score the immutable ASR hypothesis, so a corrector's effect on transcription quality is
# invisible without these. ceer_rx / corrected_ceer_rx extend CEER's critical set with the
# drug vocabulary (default CEER counts dose units only — a fixed drug name doesn't move it).
# All five are `opt_in`: the handler passes include_opt_in only when the run's
# query_correction config sets `corrected_metrics` — reports are otherwise byte-identical.
@lru_cache(maxsize=None)
def _rx_terms() -> frozenset:
    """Drug-aware CEER critical set: default units ∪ medical.json weight≥3 ∪ drug_terms."""
    from ..metrics.clinical import DEFAULT_CRITICAL_TERMS
    from ..metrics.domain_terms import load_drug_terms, load_term_weights

    terms = set(DEFAULT_CRITICAL_TERMS)
    terms |= {t for t, w in load_term_weights("medical").items() if w >= 3.0}
    terms |= set(load_drug_terms())
    # critical_entity_error_rate trusts a frozenset to be pre-lowercased
    return frozenset(t.lower() for t in terms)


register_metric(
    "corrected_wer",
    scored="corrected_query_text",
    gt="reference_transcription",
    fn=lambda hyp, ref: word_error_rate(ref, hyp),
    higher_is_better=False,
    opt_in=True,
)
register_metric(
    "corrected_cer",
    scored="corrected_query_text",
    gt="reference_transcription",
    fn=lambda hyp, ref: character_error_rate(ref, hyp),
    higher_is_better=False,
    opt_in=True,
)
register_metric(
    "corrected_ceer",
    scored="corrected_query_text",
    gt="reference_transcription",
    fn=lambda hyp, ref: critical_entity_error_rate(ref, hyp),
    higher_is_better=False,
    opt_in=True,
)
register_metric(
    "ceer_rx",
    scored="query_text",
    gt="reference_transcription",
    fn=lambda hyp, ref: critical_entity_error_rate(ref, hyp, terms=_rx_terms()),
    higher_is_better=False,
    opt_in=True,
)
register_metric(
    "corrected_ceer_rx",
    scored="corrected_query_text",
    gt="reference_transcription",
    fn=lambda hyp, ref: critical_entity_error_rate(ref, hyp, terms=_rx_terms()),
    higher_is_better=False,
    opt_in=True,
)


# IR: scored = retrieved doc-id keys (ranked list), gt = relevant_docs (doc_id → grade).
def _at_k(metric_fn: Callable[[Any, Any, int], float], k: int) -> ScoreFn:
    """Bind a rank cutoff into a (scored, gt) -> float scorer."""

    def fn(scored: Any, gt: Any) -> float:
        return float(metric_fn(scored, gt, k))

    return fn


for _k in IR_CUTOFFS:
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
