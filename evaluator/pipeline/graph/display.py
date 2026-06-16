"""Human-readable node labels for the DAG view (operator-abstraction Phase 1).

`display_label(stage, params)` turns a node's type + fields into the friendly name shown in
the graph preview, the CLI ``graph`` print and the web builder — so the *vocabulary* can
shrink to generic operators later while the *view* keeps the read-it-from-the-diagram
legibility (``DAG_OPERATOR_ABSTRACTION.md`` §4). Today the node ``stage`` is still the legacy
name; once nodes carry an operator ``stage`` + discriminator fields this resolves the label
from those fields instead (the keying is intentionally params-aware already).
"""

from __future__ import annotations

from typing import Optional

# Legacy stage name → base label (only where prettifying the snake_case name reads poorly).
_BASE_LABELS = {
    "dataset_source": "Dataset source",
    "dataset_union": "Dataset union",
    "tts": "TTS",
    "asr": "ASR",
    "text_embedding": "Text embedding",
    "audio_embedding": "Audio embedding",
    "corpus_embedding": "Corpus embedding",
    "corpus_merge": "Corpus merge",
    "vector_db": "Vector DB",
    "retrieval": "Retrieval",
    "multi_query_retrieval": "Multi-query retrieval",
    "result_fusion": "Result fusion",
    "query_correction": "Query correction",
    "query_optimization": "Query optimization",
    "query_refine": "Query refine",
    "augmenter": "Query augmenter",
    "augment_audio": "Audio augmenter",
    "rerank": "Rerank",
    "mmr": "MMR",
    "threshold": "Threshold",
    "transcription_metrics": "Transcription metrics",
    "retrieval_metrics": "Retrieval metrics",
    "answer_metrics": "Answer metrics",
    "embedding_alignment_metrics": "Embedding alignment",
    "metrics": "Report",
    "answer_gen": "Answer generation",
    "answer_judge": "LLM judge",
    "build_query_traces": "Query traces",
    "finalize": "Finalize",
    "aggregate": "Aggregate",
    "dataset_sink": "Dataset sink",
    "leaderboard_sink": "Leaderboard sink",
    "tracking_sink": "Tracking sink",
}


def _base_label(stage: str) -> str:
    return _BASE_LABELS.get(stage) or stage.replace("_", " ").capitalize()


def display_label(stage: str, params: Optional[dict] = None) -> str:
    """Friendly label for a node from its type + fields.

    Pure + deterministic. A few fields refine the base label so the diagram reads the
    experiment (a hybrid retrieval vs a dense one, an oracle ASR branch, a corpus-side
    augmenter) without the user inspecting params. ``stage`` may be an **operator** — it is
    resolved to its legacy kind (e.g. ``search{mode:hybrid}`` → ``retrieval``) so the labels
    are keyed on the concrete behavior, not the generic op name."""
    p = params or {}
    from .operators import node_kind

    stage = node_kind(stage, p)  # operator → its legacy kind (identity for legacy names)
    base = _base_label(stage)

    if stage == "retrieval":
        mode = str(p.get("mode") or "").strip()
        if mode and mode != "dense":
            return f"{mode.capitalize()} retrieval"
    if stage == "asr" and p.get("oracle"):
        return "ASR (oracle)"
    if stage == "augmenter" and p.get("axis") == "docs":
        return "Corpus augmenter"
    if stage == "vector_db":
        store = str(p.get("store") or "").strip()
        if store and store != "inmemory":
            return f"Vector DB ({store})"
    if stage in ("query_optimization", "query_refine", "rerank", "query_correction"):
        method = str(p.get("method") or p.get("mode") or "").strip()
        if method:
            return f"{base} ({method})"
    return base
