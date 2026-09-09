"""RetrievalPipeline takes strategy_config as the sole knob source.

Per-knob legacy kwargs (retrieval_mode, use_mmr, rrf_k, ...) are not accepted; None still
falls back to an all-defaults config.
"""

import pytest

from evaluator.models.retrieval.strategy import RetrievalStrategyConfig
from evaluator.pipeline.retrieval_pipeline import RetrievalPipeline
from evaluator.storage.cache import CacheManager
from evaluator.storage.vector_store import InMemoryVectorStore


def _rp(**kw):
    return RetrievalPipeline(InMemoryVectorStore(), CacheManager(enabled=False), **kw)


def test_none_strategy_config_falls_back_to_all_defaults():
    rp = _rp(strategy_config=None)
    assert rp.strategy_config == RetrievalStrategyConfig()  # dense / no-rerank / cosine
    assert rp.strategy_config.core.mode == "dense"


def test_explicit_strategy_config_is_used():
    cfg = RetrievalStrategyConfig()
    rp = _rp(strategy_config=cfg)
    assert rp.strategy_config is cfg


def test_legacy_per_knob_kwargs_are_removed():
    # legacy per-knob kwargs are not accepted — passing one is a TypeError
    with pytest.raises(TypeError):
        _rp(retrieval_mode="hybrid", use_mmr=True)
