"""Extensible artifact-type registry (architecture A11).

Artifacts flowing between nodes were a fixed enum of name constants in ``stage_graph``. To
let new fields/modalities (e.g. ``query_image``) be *additive* — register a type + an embed
op + map a datasource field to it, with no core edit — artifact types live in a registry,
like the model / node / dataset registries. Each type carries a ``modality`` and an
``is_source`` flag (source artifacts may be required without an upstream producer).

A datasource declares ``field → registered artifact`` (see :func:`validate_field_mapping`);
graph validation then works over the open, registered set instead of a closed enum.

See ``evaluator-architecture.md`` §2/§3.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Mapping

from .stage_graph import (
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
    REPORT = "report"  # the aggregated result object


@dataclass(frozen=True)
class ArtifactType:
    """A registered artifact: its ``name``, ``modality``, and whether it is a graph source."""

    name: str
    modality: Modality
    is_source: bool = False


_REGISTRY: Dict[str, ArtifactType] = {}


def register_artifact(
    name: str, modality: Modality, *, is_source: bool = False
) -> ArtifactType:
    """Register an artifact type. Re-registering the same ``(modality, is_source)`` is a
    no-op; a conflicting redefinition is an error (typo / collision protection)."""
    existing = _REGISTRY.get(name)
    new = ArtifactType(name=name, modality=modality, is_source=is_source)
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


def list_artifacts() -> List[ArtifactType]:
    return [_REGISTRY[name] for name in sorted(_REGISTRY)]


def validate_field_mapping(mapping: Mapping[str, str]) -> Dict[str, ArtifactType]:
    """Validate a datasource ``field → artifact-name`` mapping; return ``field → ArtifactType``.

    Every target artifact must be registered (else ``KeyError``). This is the O10 primitive
    that lets a datasource advertise typed fields against the open artifact vocabulary.
    """
    resolved: Dict[str, ArtifactType] = {}
    for field, artifact_name in mapping.items():
        resolved[field] = get_artifact_type(artifact_name)
    return resolved


# ── Register the built-in artifacts (the former fixed enum) ───────────
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
# Spoken-transcription GT (vs reference_text = question_text) — M1a/M1c-3.
register_artifact(ARTIFACT_REFERENCE_TRANSCRIPTION, Modality.TEXT)
register_artifact(ARTIFACT_EMBEDDING_ALIGNMENT, Modality.SCORES)
register_artifact(ARTIFACT_VECTOR_INDEX, Modality.INDEX)
register_artifact(ARTIFACT_RETRIEVED, Modality.RESULTS)
register_artifact(ARTIFACT_GENERATED_ANSWERS, Modality.ANSWERS)
register_artifact(ARTIFACT_METRICS, Modality.SCORES)
# Ground-truth reference transcription/question, published as a keyed artifact (A3/W1) so
# metric nodes can pair a scored artifact against its reference.
register_artifact("reference_text", Modality.TEXT, is_source=True)
# The query-text chain (distinct names — no in-place mutation): correction → corrected,
# augmenter → augmented, optimization → optimized. query_text stays the immutable ASR/dataset
# base. Consumers read the most-processed variant via QUERY_TEXT_CHAIN.
register_artifact("corrected_query_text", Modality.TEXT)
register_artifact("augmented_query_text", Modality.TEXT)
register_artifact("optimized_query_text", Modality.TEXT)
register_artifact("refined_query_text", Modality.TEXT)
# Per-comparison score artifacts from the typed metric nodes (Phase 5).
register_artifact("transcription_scores", Modality.SCORES)
register_artifact("retrieval_scores", Modality.SCORES)
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
