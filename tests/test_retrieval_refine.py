"""M6: similarity threshold is applied to the ORIGINAL retrieval scores, before MMR rewrites
them to λ·rel−(1−λ)·div (which isn't comparable to a similarity cutoff)."""

import numpy as np

from evaluator.models.retrieval.strategy import PostProcessingConfig, RetrievalStrategyConfig
from evaluator.pipeline.retrieval_pipeline import RetrievalPipeline
from evaluator.storage.cache import CacheManager
from evaluator.storage.vector_store import InMemoryVectorStore


def _pipeline(threshold, use_mmr):
    pp = PostProcessingConfig(
        use_mmr=use_mmr, mmr_lambda=0.5, min_similarity_threshold=threshold
    )
    cfg = RetrievalStrategyConfig(post_processing=pp)
    return RetrievalPipeline(
        InMemoryVectorStore(), CacheManager(enabled=False), strategy_config=cfg
    )


def test_threshold_filters_retrieval_score_not_mmr_score():
    rp = _pipeline(threshold=0.5, use_mmr=True)
    # two near-duplicate docs; both have high retrieval similarity to the query, but the
    # second gets a LOW mmr score (redundant with the first)
    rp.build_index(np.array([[1.0, 0.0], [0.99, 0.14]], dtype="float32"), ["d0", "d1"])
    out = rp._apply_advanced_strategies(
        np.array([1.0, 0.0], dtype="float32"), [("d0", 1.0), ("d1", 0.99)], k=5
    )
    ids = [p for p, _ in out]
    # under the fix, d1 survives (its retrieval score 0.99 > 0.5); the old order dropped it
    # because its MMR-adjusted score fell below the threshold
    assert ids == ["d0", "d1"]


def test_threshold_drops_genuinely_low_retrieval_score():
    rp = _pipeline(threshold=0.5, use_mmr=False)
    rp.build_index(np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32"), ["hit", "miss"])
    out = rp._apply_advanced_strategies(
        np.array([1.0, 0.0], dtype="float32"), [("hit", 0.95), ("miss", 0.1)], k=5
    )
    assert [p for p, _ in out] == ["hit"]  # the 0.1 retrieval is below threshold → dropped
