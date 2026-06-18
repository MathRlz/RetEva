"""IR (retrieval) diagnostics for the metrics report (extracted from ``handlers/metrics.py``).

The diagnostics the registry report does **not** carry: the WER↔Recall correlation, the
first-relevant-rank distribution + failure rate, the bootstrap confidence intervals, and (when
traces are enabled) the failure-mode decomposition. The metrics core calls ``_ir_diagnostics``
from the single-branch ``metrics`` node; values/functions are unchanged from the legacy path.
"""

from __future__ import annotations

from ...logging_config import get_logger
from ._common import _reference_transcriptions, is_asr_text_retrieval
from ...metrics import (
    first_relevant_rank_distribution,
    wer_recall_correlation,
    categorize_failures,
)
from ...metrics.ir import reciprocal_rank, ndcg_at_k
from ...analysis.errors import analyze_retrieval_failures
from ...analysis.significance import bootstrap_confidence_interval

logger = get_logger(__name__)

# Bootstrap confidence-interval settings.
BOOTSTRAP_ALPHA = 0.05
BOOTSTRAP_ITERATIONS = 1000
MIN_SAMPLES_FOR_CI = 20


def _ir_diagnostics(results, s, all_relevant, recall5, wer_scores, retrieved_keys) -> None:
    """Retrieval diagnostics not carried by the registry report (correlation, rank
    distribution, failure rate/analysis, flat CIs). Same functions/values as the old path.
    """
    corr = wer_recall_correlation(wer_scores, recall5)
    if corr is not None:
        results["wer_recall5_correlation"] = corr

    rank_dist, failure_rate = first_relevant_rank_distribution(
        retrieved_keys, all_relevant
    )
    results["first_relevant_rank_distribution"] = rank_dist
    results["retrieval_failure_rate"] = failure_rate

    if s.trace_limit > 0:
        _attach_failure_analysis(
            results, s, all_relevant, rank_dist, recall5, wer_scores, retrieved_keys
        )

    if s.compute_confidence_intervals and len(retrieved_keys) >= MIN_SAMPLES_FOR_CI:
        ci_inputs = {
            "MRR": [
                reciprocal_rank(r, rel) for r, rel in zip(retrieved_keys, all_relevant)
            ],
            "Recall@5": recall5,
            "NDCG@5": [
                ndcg_at_k(r, rel, 5) for r, rel in zip(retrieved_keys, all_relevant)
            ],
        }
        for name, ci_scores in ci_inputs.items():
            try:
                results[f"{name}_ci"] = bootstrap_confidence_interval(
                    ci_scores, alpha=BOOTSTRAP_ALPHA, n_bootstrap=BOOTSTRAP_ITERATIONS
                )
            except Exception as exc:
                logger.warning("CI computation failed for %s: %s", name, exc)


def _attach_failure_analysis(
    results, s, all_relevant, rank_dist, recall5, wer_scores, retrieved_keys
) -> None:
    """Retrieval failure-mode decomposition (only when traces are enabled)."""
    query_texts = (
        s.get_artifact("query_text", default=[])
        if is_asr_text_retrieval(s)
        else _reference_transcriptions(s)
    )
    details = [
        {
            "query": query_texts[i],
            "retrieved": retrieved_keys[i],
            "relevant": all_relevant[i],
        }
        for i in range(len(retrieved_keys))
    ]
    analysis = analyze_retrieval_failures({"details": details}, top_k=s.k)
    if wer_scores and len(wer_scores) == len(recall5):
        corpus_doc_ids = None
        if hasattr(s.dataset, "get_corpus"):
            try:
                corpus_doc_ids = {
                    str(doc.get("doc_id", "")) for doc in s.dataset.get_corpus()
                }
                corpus_doc_ids.discard("")
            except Exception as exc:
                logger.debug("corpus doc-id extraction failed: %s", exc)
                corpus_doc_ids = None
        analysis["failure_categories"] = categorize_failures(
            wer_scores,
            recall5,
            rank_dist,
            all_relevant=all_relevant,
            corpus_doc_ids=corpus_doc_ids,
        )
    results["retrieval_failure_analysis"] = analysis
