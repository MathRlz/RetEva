"""Metric-spec registry + per-item computation. A metric declares the artifacts it needs (a
*scored* artifact + optional *ground-truth* artifact); the builder/runtime auto-injects it
wherever those inputs are present. :func:`compute_metric` aligns scored/GT ``ItemSet``s by id;
reducing to scalars and cross-branch deltas is ``aggregate.py``'s job."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from ..metrics.clinical import critical_entity_error_rate
from ..constants import IR_CUTOFFS
from ..metrics.ir import (
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
    opt_in: bool = False  # computed only when the caller passes include_opt_in

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
    """Run every applicable metric over the available keyed ``artifacts`` -> ``{metric_name:
    item_scores}``. A per-branch metrics node calls this; ``aggregate`` then reduces the
    returned ``item_scores`` to scalars + cross-branch deltas.

    ``only`` (the config's ``metrics:`` allowlist) computes exactly those registered names —
    explicit naming bypasses ``opt_in``; a name whose inputs aren't present is skipped
    (unknown names raise via :func:`get_metric`, but config validation rejects them
    earlier). ``None`` keeps collect-all."""
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


# ASR: scored = query_text (immutable ASR hypothesis); gt = reference_transcription (the
# SPOKEN transcription, distinct from the question-text bus artifact). wer/cer always
# measure ASR quality; a "corrected WER" is expressible by pointing a second metric at
# corrected_query_text.
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
# Targets the actually-dangerous entities rather than up-weighting a broad medical vocabulary.
register_metric(
    "ceer",
    scored="query_text",
    gt="reference_transcription",
    fn=lambda hyp, ref: critical_entity_error_rate(ref, hyp),
    higher_is_better=False,
)


# corrected_* score the CORRECTED text against the reference — the default wer/cer/ceer
# score the immutable ASR hypothesis, so a corrector's effect would otherwise be invisible.
# ceer_rx/corrected_ceer_rx extend CEER's critical set with drug vocabulary (default CEER
# counts dose units only). All five are `opt_in`: enabled only when the run's
# query_correction config sets `corrected_metrics` — reports are otherwise byte-identical.
@lru_cache(maxsize=None)
def _rx_terms() -> frozenset:
    """Drug-aware CEER critical set: default units ∪ medical.json weight≥3 ∪ drug_terms."""

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
# Mean average precision (per-query AP; the run mean is MAP).
register_metric(
    "map",
    scored="retrieved",
    gt="relevant_docs",
    fn=lambda ret, rel: float(average_precision(ret, rel)),
)

# The judge publishes per-query numeric ItemSets; each metric scores its own artifact, so it
# auto-injects only when present (only configured aspects fire). fn is the identity — the
# score is already in [0, 1] (pass is 1.0/0.0 → its mean is the pass rate).
from ..config.judge import VALID_JUDGE_ASPECTS  # noqa: E402
from ..metrics.clinical import DEFAULT_CRITICAL_TERMS
from ..metrics.domain_terms import load_drug_terms, load_term_weights

register_metric("judge_overall", scored="judge_scores", gt=None, fn=lambda v, _g: float(v))
register_metric("judge_pass_rate", scored="judge_pass", gt=None, fn=lambda v, _g: float(v))
for _aspect in sorted(VALID_JUDGE_ASPECTS):
    register_metric(
        f"judge_{_aspect}", scored=f"judge_aspect_{_aspect}", gt=None,
        fn=lambda v, _g: float(v),
    )
