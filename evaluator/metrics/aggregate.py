"""Aggregate metric helpers for retrieval evaluation."""

from typing import Dict, List

from .ir import reciprocal_rank, average_precision, recall_at_k, ndcg_at_k


def compute_ir_metrics(
    all_retrieved: List[List[str]],
    all_relevant: List[Dict[str, int]],
    k_values: List[int] = None,
) -> Dict[str, float]:
    """Compute aggregated IR metrics from retrieved and relevant items."""
    if k_values is None:
        k_values = [1, 5, 10]

    recall_at_ks = {k_val: [] for k_val in k_values}
    ndcg_at_ks = {k_val: [] for k_val in k_values}
    rr = []
    precision = []

    for retrieved, relevant in zip(all_retrieved, all_relevant):
        for k_val in k_values:
            recall = recall_at_k(retrieved, relevant, k_val)
            recall_at_ks[k_val].append(recall)

            ndcg = ndcg_at_k(retrieved, relevant, k_val)
            ndcg_at_ks[k_val].append(ndcg)

        rr.append(reciprocal_rank(retrieved, relevant))
        precision.append(average_precision(retrieved, relevant))

    results: Dict[str, float] = {}
    results["MRR"] = sum(rr) / len(rr) if len(rr) > 0 else 0.0
    results["MAP"] = sum(precision) / len(precision) if len(precision) > 0 else 0.0

    for kk in k_values:
        results[f"Recall@{kk}"] = (
            sum(recall_at_ks[kk]) / len(recall_at_ks[kk])
            if len(recall_at_ks[kk]) > 0 else 0.0
        )
        results[f"NDCG@{kk}"] = (
            sum(ndcg_at_ks[kk]) / len(ndcg_at_ks[kk])
            if len(ndcg_at_ks[kk]) > 0 else 0.0
        )

    return results


def log_ir_metrics(results: Dict[str, float], logger, k_values: List[int] = None):
    """Log IR metrics in a formatted way."""
    if k_values is None:
        k_values = [1, 5, 10]

    logger.info("IR Metrics:")
    logger.info(f"  MRR: {results['MRR']:.4f}")
    logger.info(f"  MAP: {results['MAP']:.4f}")
    for kk in k_values:
        logger.info(f"  Recall@{kk}: {results[f'Recall@{kk}']:.4f}")
        logger.info(f"  NDCG@{kk}: {results[f'NDCG@{kk}']:.4f}")

