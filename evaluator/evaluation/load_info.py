"""Typed carrier for the run's dataset-load channel (was an untyped shared dict).

One instance rides ``RunState.load_info`` per run, shared by reference between the
service wrapper and the in-graph handlers: ``replay_query_ids`` goes *in* (item replay
slices the dataset at load), ``dataset`` comes *out* (the wrapper's num_samples needs no
pre-graph load), and the corpus-index path stamps its vector-cache outcome on it.
"""

from dataclasses import dataclass, fields
from typing import Any, Dict, List, Optional


@dataclass
class LoadInfo:
    replay_query_ids: Optional[List[str]] = None
    dataset: Any = None
    corpus_size: Optional[int] = None
    vector_cache_hit: Optional[bool] = None
    vector_cache_key: Optional[str] = None
    vector_cache_written: Optional[bool] = None

    def to_metadata(self) -> Dict[str, Any]:
        """The report-metadata dict (``metadata.cache.load``): only the fields that were
        actually set, matching the shape of the former plain dict."""
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if getattr(self, f.name) is not None
        }
