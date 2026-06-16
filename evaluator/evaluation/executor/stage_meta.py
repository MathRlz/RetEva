"""Per-model-stage placement metadata — one source of truth for the two executor maps.

A model stage needs two things keyed by its legacy kind: which ``config.model`` field holds its
device (the parallel scheduler serializes same-device work) and which ``RunState`` attribute holds
the pipeline whose model it runs (the offload planner frees it after its last use). These used to
live as two hand-kept dicts in ``parallel.py`` and ``offload.py``; they're derived from this table
instead so adding a model stage is a single edit.

``corpus_embedding`` carries a device (it shares the query text embedder's) but no own pipeline
attr — the offload planner resolves its model specially — so its ``pipeline_attr`` is ``None``.
"""

from __future__ import annotations

from typing import Dict, Optional

# legacy stage kind -> (config.model device field, RunState pipeline attr)
STAGE_META: Dict[str, Dict[str, Optional[str]]] = {
    "asr": {"device_attr": "asr_device", "pipeline_attr": "asr_pipeline"},
    "text_embedding": {
        "device_attr": "text_emb_device",
        "pipeline_attr": "text_embedding_pipeline",
    },
    "audio_embedding": {
        "device_attr": "audio_emb_device",
        "pipeline_attr": "audio_embedding_pipeline",
    },
    "corpus_embedding": {"device_attr": "text_emb_device", "pipeline_attr": None},
}

# Derived projections used by the executor (None entries dropped).
DEVICE_ATTR: Dict[str, str] = {
    k: v["device_attr"] for k, v in STAGE_META.items() if v["device_attr"]
}
PIPELINE_ATTR: Dict[str, str] = {
    k: v["pipeline_attr"] for k, v in STAGE_META.items() if v["pipeline_attr"]
}
