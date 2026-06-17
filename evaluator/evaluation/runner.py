"""Shared config→results execution entrypoint.

Thin alias over the single execution core in ``services.evaluation_service._run_core`` so
the CLI, the webapi, and the public API all run exactly one path (descriptor-based
loading, audio synthesis, corpus build + vector-DB cache, evaluate). Config-first: the
full config is the experiment. The caller owns logging, result saving, leaderboard
ingest, and any model-service lifecycle.
"""

from typing import Any, Callable, Dict, Optional

from evaluator.config import EvaluationConfig
from evaluator.storage.cache import CacheManager


def run_evaluation_from_config(
    config: EvaluationConfig,
    *,
    cache_manager: Optional[CacheManager] = None,
    progress_callback: Optional[Callable[..., None]] = None,
    service_provider: Any = None,
    query_ids: Optional[Any] = None,
) -> Dict[str, Any]:
    """Build pipelines, prepare the dataset + corpus, and evaluate. Returns results dict.

    ``query_ids`` (Roadmap 2d) slices the query side to those ids before evaluating — the
    item-replay path; the corpus stays whole."""
    from evaluator.services.evaluation_service import _run_core

    if cache_manager is None:
        cache_manager = CacheManager(
            cache_dir=config.cache.cache_dir, enabled=config.cache.enabled
        )
    metrics, _dataset = _run_core(
        config,
        cache_manager=cache_manager,
        service_provider=service_provider,
        progress_callback=progress_callback,
        query_ids=query_ids,
    )
    return metrics
