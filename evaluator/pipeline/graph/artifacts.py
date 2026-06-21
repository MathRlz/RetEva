"""Artifact vocabulary for the stage-graph DAG (Data-as-DAG phase 1).

The named, typed values passed between nodes, plus the ``OneOf`` priority-input
primitive. Edges connect a producer's output to a consumer's input *by artifact name*.
``SOURCE_ARTIFACTS`` are provided externally by the dataset / runtime (so a node can
require them without an upstream producer).

This module is pure data + the ``OneOf`` primitives; it has no dependency on
``registry`` (which re-exports these names so they remain importable from there).
"""

from typing import Any, Dict, List, Optional, Tuple


class OneOf(tuple):
    """An input that binds to the highest-priority *available* alternative.

    Entries are artifact names in priority order (highest first). At wiring time the
    consumer binds to the first alternative that has an upstream producer; the handler
    reads it under the canonical ``key`` (default: the last/base name) via ``s.input(key)``.

    This replaces *newest-producer-wins* resolution with an explicit, execution-order
    independent priority, so distinctly-named transform outputs (e.g.
    ``optimized_query_text`` → ``corrected_query_text`` → ``query_text``) chain without
    any in-place artifact mutation.
    """

    def __new__(cls, names: Tuple[str, ...], key: Optional[str] = None) -> "OneOf":
        obj = super().__new__(cls, tuple(names))
        obj.key = key if key is not None else (obj[-1] if len(obj) else None)
        return obj


def one_of(*names: str, key: Optional[str] = None) -> OneOf:
    """Ordered input alternatives (highest priority first). See :class:`OneOf`."""
    return OneOf(names, key=key)


def display_artifact_names(seq: Tuple[Any, ...]) -> Tuple[str, ...]:
    """Flatten an inputs/optional_inputs tuple for display: an :class:`OneOf` expands to
    its alternatives, plain names pass through. Used by form/graph introspection so a
    ``OneOf`` entry never reaches a ``", ".join`` as a nested tuple."""
    out: List[str] = []
    for art in seq:
        if isinstance(art, OneOf):
            out.extend(art)
        else:
            out.append(art)
    return tuple(out)


# ── Artifact vocabulary (Data-as-DAG phase 1) ─────────────────────────
# Named, typed values passed between nodes. Edges connect a producer's output to a
# consumer's input *by artifact name*. SOURCE_ARTIFACTS are provided externally by the
# dataset / runtime (so a node can require them without an upstream producer).
#
# ``query_text`` is the IMMUTABLE base query: the dataset's question text (text modes) or
# the ASR hypothesis (audio modes). It is NEVER overwritten — the correction / augmentation /
# optimization transforms each emit a DISTINCT name (corrected/augmented/optimized) and
# downstream consumers read the chain via ``QUERY_TEXT_CHAIN``. Because query_text is the
# un-rewritten ASR hypothesis, WER/CER score directly against it (no raw_query_text needed).
ARTIFACT_QUERY_AUDIO = "query_audio"
ARTIFACT_QUERY_TEXT = "query_text"
ARTIFACT_CORRECTED_QUERY_TEXT = "corrected_query_text"
ARTIFACT_AUGMENTED_QUERY_TEXT = "augmented_query_text"
ARTIFACT_OPTIMIZED_QUERY_TEXT = "optimized_query_text"
# Post-retrieval reformulation (query_refine): the query improved using retrieved docs as
# context — the top of the chain (most-processed). Drives iterative RAG: a later hop's
# text_embedding reads this over the pre-retrieval variants.
ARTIFACT_REFINED_QUERY_TEXT = "refined_query_text"
# The "current query text" every query-text consumer reads: the most-processed variant a
# bound producer actually published (optimized > augmented > corrected > the base ASR /
# dataset query_text). Shared by correction/augmenter/optimization/text_embedding/
# retrieval/rerank — wiring restricts each node to the variants produced upstream of it,
# so the same chain expresses every correction/optimization on/off combination with no
# in-place mutation. Read via ``s.input("query_text")``.
QUERY_TEXT_CHAIN = one_of(
    ARTIFACT_REFINED_QUERY_TEXT,
    ARTIFACT_OPTIMIZED_QUERY_TEXT,
    ARTIFACT_AUGMENTED_QUERY_TEXT,
    ARTIFACT_CORRECTED_QUERY_TEXT,
    ARTIFACT_QUERY_TEXT,
    key=ARTIFACT_QUERY_TEXT,
)
# The spoken-transcription ground truth (dataset "transcription" field) — distinct from
# reference_text (= question_text): they coincide only on TTS-bridge datasets where the
# spoken text IS the question (M1a/M1c-3). ASR-quality metrics score against THIS.
ARTIFACT_REFERENCE_TRANSCRIPTION = "reference_transcription"
ARTIFACT_CORPUS = "corpus"
ARTIFACT_RELEVANT_DOCS = "relevant_docs"
ARTIFACT_SHORT_ANSWERS = "short_answers"
ARTIFACT_QUERY_VECTORS = "query_vectors"
# Distinct per-stream query embeddings (no in-place query_vectors mutation). The retrieval
# input is one_of(fused, audio, text, query_vectors): a consumer names what it wants and
# the runtime falls back across them (e.g. fusion bails -> audio).
ARTIFACT_AUDIO_QUERY_VECTORS = "audio_query_vectors"
ARTIFACT_TEXT_QUERY_VECTORS = "text_query_vectors"
ARTIFACT_FUSED_QUERY_VECTORS = "fused_query_vectors"
# Embedded corpus (vectors + aligned payloads + space tag) — the artifact between the
# split corpus_embedding and vector_db nodes (§4 split).
ARTIFACT_CORPUS_VECTORS = "corpus_vectors"
ARTIFACT_VECTOR_INDEX = "vector_index"
ARTIFACT_RETRIEVED = "retrieved"
# Audio<->text cosine-alignment diagnostic, published by the fusion node (M1d-2).
ARTIFACT_EMBEDDING_ALIGNMENT = "embedding_alignment"
ARTIFACT_METRICS = "metrics"
ARTIFACT_GENERATED_ANSWERS = "generated_answers"
# Per-comparison score artifacts, published by the typed metric nodes (Phase 5). Each
# pairs an EXPECTED value (dataset_source GT) against an ACTUAL value (a transform's
# output): transcription_metrics = reference_transcription × query_text;
# retrieval_metrics = relevant_docs × retrieved. The `metrics` node assembles the report
# from them (+ the per-item RunState intermediates they set).
ARTIFACT_TRANSCRIPTION_SCORES = "transcription_scores"
ARTIFACT_RETRIEVAL_SCORES = "retrieval_scores"
# LLM-judge verdicts (answer_judge node): generated answer / retrieval × the judge rubric.
# judge_scores = overall per-query score; judge_pass = per-query 1.0/0.0 (→ judge_pass_rate).
# Both are always published; the per-aspect judge_aspect_* outputs depend on config and so
# are not part of the static node contract.
ARTIFACT_JUDGE_SCORES = "judge_scores"
ARTIFACT_JUDGE_PASS = "judge_pass"
# Per-query traces (built by the explicit build_query_traces node; read by judge + report).
ARTIFACT_QUERY_TRACES = "query_traces"
# Generated-answer scores (answer_metrics node): reference_answer × generated_answer.
ARTIFACT_ANSWER_SCORES = "answer_scores"

SOURCE_ARTIFACTS = frozenset(
    {
        ARTIFACT_QUERY_AUDIO,
        ARTIFACT_QUERY_TEXT,
        ARTIFACT_CORPUS,
        ARTIFACT_RELEVANT_DOCS,
        ARTIFACT_SHORT_ANSWERS,
    }
)

# ── dataset_source roles (multi-dataset graphs) ───────────────────────
# A dataset_source node's ``role`` param narrows what it advertises, so a graph can mix
# multiple sources (e.g. corpus from dataset A + questions from dataset B) and downstream
# nodes auto-wire to the *right* source by artifact name. ``both`` (default) advertises
# everything = single-dataset back-compat.
DATASET_ROLE_CORPUS = "corpus"
DATASET_ROLE_QUESTIONS = "questions"
DATASET_ROLE_BOTH = "both"

# Type-preserving T→T nodes that support the corpus axis (`axis: docs`, §4.1 T2).
# Extend as experiments need it (query_correction / query_optimization are candidates).
_DOCS_AXIS_CAPABLE = frozenset({"augmenter"})

_DATASET_SOURCE_ROLE_OUTPUTS: Dict[str, Tuple[str, ...]] = {
    DATASET_ROLE_CORPUS: (ARTIFACT_CORPUS,),
    DATASET_ROLE_QUESTIONS: (
        ARTIFACT_QUERY_AUDIO,
        ARTIFACT_QUERY_TEXT,
        ARTIFACT_RELEVANT_DOCS,
        ARTIFACT_SHORT_ANSWERS,
        ARTIFACT_REFERENCE_TRANSCRIPTION,
    ),
}
