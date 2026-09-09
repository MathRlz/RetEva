"""Extensible artifact-type registry.

New fields/modalities register additively (type + embed op + datasource field map, no core
edit) instead of a fixed enum; each type carries a ``modality`` and ``is_source`` flag."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Mapping, Optional

from .graph.artifacts import (
    ARTIFACT_CORPUS,
    ARTIFACT_CORPUS_VECTORS,
    ARTIFACT_GENERATED_ANSWERS,
    ARTIFACT_METRICS,
    ARTIFACT_QUERY_AUDIO,
    ARTIFACT_QUERY_TEXT,
    ARTIFACT_QUERY_VECTORS,
    ARTIFACT_AUDIO_QUERY_VECTORS,
    ARTIFACT_TEXT_QUERY_VECTORS,
    ARTIFACT_FUSED_QUERY_VECTORS,
    ARTIFACT_REFERENCE_TRANSCRIPTION,
    ARTIFACT_EMBEDDING_ALIGNMENT,
    ARTIFACT_RELEVANT_DOCS,
    ARTIFACT_RETRIEVED,
    ARTIFACT_SHORT_ANSWERS,
    ARTIFACT_VECTOR_INDEX,
)


class Modality(str, Enum):
    """The kind of data an artifact carries (drives which transform ops can consume it)."""

    TEXT = "text"
    AUDIO = "audio"
    IMAGE = "image"
    VECTOR = "vector"
    DOC_SET = "doc_set"  # the corpus (a set of typed records)
    RELEVANCE = "relevance"  # query↔doc grades (ground truth)
    ANSWERS = "answers"  # reference or generated answers
    RESULTS = "results"  # ranked retrieval results
    INDEX = "index"  # a searchable vector index
    SCORES = "scores"  # per-item metric scores


# Human/dev-facing concrete shape per modality (not runtime-enforced), the default ``dtype``
# a registered artifact gets when it doesn't need a more precise override. A dtype is a
# property of the ARTIFACT, not of the dataset, so this stays a flat modality-keyed table.
MODALITY_DTYPES: Dict[Modality, str] = {
    Modality.TEXT: "str",
    Modality.AUDIO: "AudioRef",
    Modality.IMAGE: "ImageRef",
    Modality.VECTOR: "List[float32]",
    Modality.DOC_SET: "List[CorpusDocument]",
    Modality.RELEVANCE: "Dict[str, int]",
    Modality.ANSWERS: "List[str]",
    Modality.RESULTS: "List[Tuple[str, float]]",
    Modality.INDEX: "RetrievalPipeline",
    Modality.SCORES: "float",
}


@dataclass(frozen=True)
class ArtifactType:
    """A registered artifact: its ``name``, ``modality``, whether it is a graph source,
    and its ``dtype`` — a human/dev-facing concrete shape (e.g. ``"List[CorpusDocument]"``),
    defaulted from ``modality`` via :data:`MODALITY_DTYPES` when not given explicitly."""

    name: str
    modality: Modality
    is_source: bool = False
    dtype: Optional[str] = None


_REGISTRY: Dict[str, ArtifactType] = {}


def register_artifact(
    name: str, modality: Modality, *, is_source: bool = False, dtype: Optional[str] = None
) -> ArtifactType:
    """Register an artifact type; ``dtype`` defaults to :data:`MODALITY_DTYPES` unless given
    explicitly. Re-registering the same ``(modality, is_source, dtype)`` is a no-op; a
    conflicting redefinition raises (typo/collision protection)."""
    existing = _REGISTRY.get(name)
    resolved_dtype = dtype or MODALITY_DTYPES.get(modality)
    new = ArtifactType(name=name, modality=modality, is_source=is_source, dtype=resolved_dtype)
    if existing is not None and existing != new:
        raise ValueError(
            f"artifact '{name}' already registered as {existing}; cannot redefine to {new}"
        )
    _REGISTRY[name] = new
    return new


def get_artifact_type(name: str) -> ArtifactType:
    if name not in _REGISTRY:
        known = ", ".join(sorted(_REGISTRY))
        raise KeyError(f"unknown artifact '{name}'. Registered: {known}")
    return _REGISTRY[name]


def is_registered(name: str) -> bool:
    return name in _REGISTRY


def artifact_modality(name: str) -> Modality:
    return get_artifact_type(name).modality


def artifact_dtype(name: str) -> str:
    return get_artifact_type(name).dtype


def list_artifacts() -> List[ArtifactType]:
    return [_REGISTRY[name] for name in sorted(_REGISTRY)]


def validate_field_mapping(mapping: Mapping[str, str]) -> Dict[str, ArtifactType]:
    """Validate a datasource ``field -> artifact-name`` mapping; return ``field -> ArtifactType``,
    letting a datasource advertise typed fields against the open artifact vocabulary."""
    resolved: Dict[str, ArtifactType] = {}
    for field, artifact_name in mapping.items():
        resolved[field] = get_artifact_type(artifact_name)
    return resolved


# Source artifacts (provided by dataset_source; may be required without a producer).
register_artifact(ARTIFACT_QUERY_TEXT, Modality.TEXT, is_source=True)
register_artifact(ARTIFACT_QUERY_AUDIO, Modality.AUDIO, is_source=True)
register_artifact(ARTIFACT_CORPUS, Modality.DOC_SET, is_source=True)
register_artifact(ARTIFACT_RELEVANT_DOCS, Modality.RELEVANCE, is_source=True)
register_artifact(ARTIFACT_SHORT_ANSWERS, Modality.ANSWERS, is_source=True)
# Derived artifacts (produced by transform / retrieval / metric nodes).
register_artifact(ARTIFACT_QUERY_VECTORS, Modality.VECTOR)
register_artifact(ARTIFACT_AUDIO_QUERY_VECTORS, Modality.VECTOR)
register_artifact(ARTIFACT_TEXT_QUERY_VECTORS, Modality.VECTOR)
register_artifact(ARTIFACT_FUSED_QUERY_VECTORS, Modality.VECTOR)
register_artifact(ARTIFACT_CORPUS_VECTORS, Modality.VECTOR)
# Spoken-transcription GT (vs reference_text = question_text).
register_artifact(ARTIFACT_REFERENCE_TRANSCRIPTION, Modality.TEXT)
register_artifact(ARTIFACT_EMBEDDING_ALIGNMENT, Modality.SCORES)
register_artifact(ARTIFACT_VECTOR_INDEX, Modality.INDEX)
# Per-query ranked hits, not a single ranked list — override the RESULTS default.
register_artifact(ARTIFACT_RETRIEVED, Modality.RESULTS, dtype="List[List[Tuple[str, float]]]")
register_artifact(ARTIFACT_GENERATED_ANSWERS, Modality.ANSWERS)
register_artifact(ARTIFACT_METRICS, Modality.SCORES)
# Published as a keyed artifact so metric nodes can pair a scored artifact against it.
register_artifact("reference_text", Modality.TEXT, is_source=True)
# Distinct chain names (no in-place mutation); consumers read the most-processed variant
# via QUERY_TEXT_CHAIN, query_text stays the immutable ASR/dataset base.
register_artifact("corrected_query_text", Modality.TEXT)
register_artifact("augmented_query_text", Modality.TEXT)
register_artifact("optimized_query_text", Modality.TEXT)
register_artifact("refined_query_text", Modality.TEXT)
# Per-comparison score artifacts from the typed metric nodes.
register_artifact("transcription_scores", Modality.SCORES)
register_artifact("retrieval_scores", Modality.SCORES)
# Per-item score ItemSets: published keyed, joined by query id — replacing positional
# RunState side-channel attrs.
register_artifact("per_query_wer", Modality.SCORES)
register_artifact("per_query_cer", Modality.SCORES)
register_artifact("per_query_recall5", Modality.SCORES)
register_artifact("generated_answers", Modality.ANSWERS)
register_artifact("judge_scores", Modality.SCORES)  # overall per-query judge score
register_artifact("judge_pass", Modality.SCORES)  # per-query 1.0/0.0 → judge_pass_rate
# per-aspect judge scores; only the configured aspects are actually published at run time
for _judge_aspect in (
    "relevance", "faithfulness", "correctness", "completeness", "clarity", "accuracy", "factuality",
):
    register_artifact(f"judge_aspect_{_judge_aspect}", Modality.SCORES)
register_artifact("answer_scores", Modality.SCORES)
# Planned modalities are additive — e.g. an image query field:
#   register_artifact("query_image", Modality.IMAGE, is_source=True)
