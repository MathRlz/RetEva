"""A5 pins (CRITIQUE.md): metrics/ir.py matches the trec_eval reference implementation.

The IR metric layer is hand-rolled; this cross-checks every registry-served IR metric
(recall@k / precision@k / ndcg@k / mrr / map) against ``pytrec_eval`` (the trec_eval
bindings) per query to 1e-9 on a fixture covering graded relevance, judged-non-relevant
docs, unjudged retrieved docs, unretrieved relevant docs, and short/deep rankings.
Dev-only dependency (``pip install -e .[dev]``); skipped when absent.
"""

import pytest

pytrec_eval = pytest.importorskip("pytrec_eval")

from evaluator.metrics.ir import (  # noqa: E402
    average_precision,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

# Graded qrels: 0 = judged non-relevant (must not count as relevant anywhere).
QRELS = {
    "q1": {"d1": 2, "d2": 1, "d3": 0, "d9": 3},  # graded; d9 relevant but never retrieved
    "q2": {"d4": 1},                              # single relevant, retrieved at rank 3
    "q3": {"d7": 1, "d8": 2},                     # relevant docs entirely absent from the run
    "q4": {"d1": 1, "d2": 1, "d3": 1, "d4": 1, "d5": 1, "d6": 1},  # more relevant than k=5
    "q5": {"d5": 3},                              # top-1 hit, high grade
}

# Ranked runs (best first); includes unjudged docs at every depth.
RUNS = {
    "q1": ["d1", "d3", "d2", "d4", "d5", "d6", "d7", "d8", "d10", "d11", "d12"],
    "q2": ["d1", "d2", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11"],
    "q3": ["d1", "d2", "d3", "d4", "d5", "d6", "d9", "d10", "d11", "d12"],
    "q4": ["d6", "d10", "d1", "d11", "d2", "d3", "d12", "d4", "d13", "d5"],
    "q5": ["d5", "d1", "d2"],                     # shorter than k=5/10
}

K_VALUES = (1, 5, 10)


@pytest.fixture(scope="module")
def trec_results():
    run = {
        qid: {doc: float(len(docs) - i) for i, doc in enumerate(docs)}
        for qid, docs in RUNS.items()
    }
    measures = {"map", "recip_rank"}
    measures |= {f"P_{k}" for k in K_VALUES}
    measures |= {f"ndcg_cut_{k}" for k in K_VALUES}
    measures |= {f"recall_{k}" for k in K_VALUES}
    evaluator = pytrec_eval.RelevanceEvaluator(QRELS, measures)
    return evaluator.evaluate(run)


@pytest.mark.parametrize("qid", sorted(QRELS))
@pytest.mark.parametrize("k", K_VALUES)
def test_recall_matches_trec_eval(trec_results, qid, k):
    ours = recall_at_k(RUNS[qid], QRELS[qid], k)
    assert ours == pytest.approx(trec_results[qid][f"recall_{k}"], abs=1e-9)


@pytest.mark.parametrize("qid", sorted(QRELS))
@pytest.mark.parametrize("k", K_VALUES)
def test_precision_matches_trec_eval(trec_results, qid, k):
    ours = precision_at_k(RUNS[qid], QRELS[qid], k)
    assert ours == pytest.approx(trec_results[qid][f"P_{k}"], abs=1e-9)


@pytest.mark.parametrize("qid", sorted(QRELS))
@pytest.mark.parametrize("k", K_VALUES)
def test_ndcg_matches_trec_eval(trec_results, qid, k):
    ours = ndcg_at_k(RUNS[qid], QRELS[qid], k)
    assert ours == pytest.approx(trec_results[qid][f"ndcg_cut_{k}"], abs=1e-9)


@pytest.mark.parametrize("qid", sorted(QRELS))
def test_reciprocal_rank_matches_trec_eval(trec_results, qid):
    ours = reciprocal_rank(RUNS[qid], QRELS[qid])
    assert ours == pytest.approx(trec_results[qid]["recip_rank"], abs=1e-9)


@pytest.mark.parametrize("qid", sorted(QRELS))
def test_average_precision_matches_trec_eval(trec_results, qid):
    ours = average_precision(RUNS[qid], QRELS[qid])
    assert ours == pytest.approx(trec_results[qid]["map"], abs=1e-9)
