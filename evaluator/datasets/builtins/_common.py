"""Pieces shared by more than one built-in dataset's ``from_config``."""
from __future__ import annotations

from typing import Any, TYPE_CHECKING
from ..runtime import AudioSamplesQueryDataset, _load_corpus_entries

if TYPE_CHECKING:
    from ...config.data import DataConfig
    from ..core import QueryDataset


def audio_query_dataset(loader: Any, data: "DataConfig") -> "QueryDataset":
    """Run ``loader`` and wrap its samples, attaching the corpus at ``data.corpus_path`` when
    one is configured — the tail every corpus-backed audio-query built-in shares once it has its
    loader."""

    samples = loader.load()
    corpus_entries = (
        _load_corpus_entries(data.corpus_path)
        if getattr(data, "corpus_path", None)
        else []
    )
    return AudioSamplesQueryDataset(
        samples,
        trace_limit=getattr(data, "trace_limit", 0),
        corpus_entries=corpus_entries,
    )
