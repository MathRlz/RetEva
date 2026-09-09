"""Artifact vocabulary for the stage-graph DAG.

Named, typed values passed between nodes (connected by artifact name) + the ``OneOf``
priority-input primitive; pure data, no dependency on ``registry`` (which re-exports these)."""

from typing import Any, Dict, List, Optional, Tuple


class OneOf(tuple):
    """An input that binds to the highest-priority *available* alternative: names in priority
    order (highest first); the handler reads it under the canonical ``key`` via ``s.input(key)``.
    Replaces newest-producer-wins with an explicit, order-independent priority, so
    distinctly-named transform outputs chain without in-place artifact mutation."""

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


# ``query_text`` is IMMUTABLE (never overwritten): the dataset's question text or the ASR
# hypothesis. Correction/augmentation/optimization each emit a DISTINCT name instead, read
# via ``QUERY_TEXT_CHAIN``; WER/CER score directly against this un-rewritten hypothesis.
ARTIFACT_QUERY_AUDIO = "query_audio"
ARTIFACT_QUERY_TEXT = "query_text"
ARTIFACT_CORRECTED_QUERY_TEXT = "corrected_query_text"
ARTIFACT_AUGMENTED_QUERY_TEXT = "augmented_query_text"
ARTIFACT_OPTIMIZED_QUERY_TEXT = "optimized_query_text"
# Post-retrieval reformulation (query_refine): the top of the chain (most-processed); drives
# iterative RAG, a later hop's text_embedding reads this over pre-retrieval variants.
ARTIFACT_REFINED_QUERY_TEXT = "refined_query_text"
# The "current query text" every consumer reads: the most-processed variant a bound producer
# actually published (optimized > augmented > corrected > base). Wiring restricts each node
# to variants produced upstream of it, so one chain expresses every on/off combination with
# no in-place mutation.
QUERY_TEXT_CHAIN = one_of(
    ARTIFACT_REFINED_QUERY_TEXT,
    ARTIFACT_OPTIMIZED_QUERY_TEXT,
    ARTIFACT_AUGMENTED_QUERY_TEXT,
    ARTIFACT_CORRECTED_QUERY_TEXT,
    ARTIFACT_QUERY_TEXT,
    key=ARTIFACT_QUERY_TEXT,
)
# Spoken-transcription GT, distinct from reference_text (=question_text): they coincide only
# on TTS-bridge datasets where the spoken text IS the question. ASR metrics score against this.
ARTIFACT_REFERENCE_TRANSCRIPTION = "reference_transcription"
# The retrieval-side reference (=question text), published by the source; declared here so
# it's a real, bindable output like every other GT artifact.
ARTIFACT_REFERENCE_TEXT = "reference_text"
ARTIFACT_CORPUS = "corpus"
ARTIFACT_RELEVANT_DOCS = "relevant_docs"
ARTIFACT_SHORT_ANSWERS = "short_answers"
ARTIFACT_QUERY_VECTORS = "query_vectors"
# Distinct per-stream query embeddings; retrieval's one_of(fused, audio, text, query_vectors)
# falls back across them (e.g. fusion bails -> audio) with no in-place mutation.
ARTIFACT_AUDIO_QUERY_VECTORS = "audio_query_vectors"
ARTIFACT_TEXT_QUERY_VECTORS = "text_query_vectors"
ARTIFACT_FUSED_QUERY_VECTORS = "fused_query_vectors"
# Embedded corpus (vectors + aligned payloads + space tag) — the artifact between the
# split corpus_embedding and vector_db nodes.
ARTIFACT_CORPUS_VECTORS = "corpus_vectors"
ARTIFACT_VECTOR_INDEX = "vector_index"
ARTIFACT_RETRIEVED = "retrieved"
# Audio<->text cosine-alignment diagnostic, published by the fusion node.
ARTIFACT_EMBEDDING_ALIGNMENT = "embedding_alignment"
ARTIFACT_METRICS = "metrics"
ARTIFACT_GENERATED_ANSWERS = "generated_answers"
# Per-comparison score artifacts from the typed metric nodes, each pairing an EXPECTED value
# (dataset_source GT) against an ACTUAL value: transcription_metrics = reference_transcription
# x query_text; retrieval_metrics = relevant_docs x retrieved. `metrics` assembles the report.
ARTIFACT_TRANSCRIPTION_SCORES = "transcription_scores"
ARTIFACT_RETRIEVAL_SCORES = "retrieval_scores"
# LLM-judge verdicts: judge_scores = overall per-query score; judge_pass = per-query 1.0/0.0
# (-> judge_pass_rate). Both always published; per-aspect judge_aspect_* depend on config.
ARTIFACT_JUDGE_SCORES = "judge_scores"
ARTIFACT_JUDGE_PASS = "judge_pass"
# Per-query traces (built by the explicit build_query_traces node; read by judge + report).
ARTIFACT_QUERY_TRACES = "query_traces"
# Generated-answer scores (answer_metrics node): reference_answer × generated_answer.
ARTIFACT_ANSWER_SCORES = "answer_scores"

# Artifacts the simplified UI hides from a meaningful node's input ports when unproduced:
# their producer is always-hidden reporting plumbing, not a normal shown node (unlike e.g.
# corpus_vectors, whose producer corpus_embedding is shown).
STRUCTURAL_ARTIFACTS = frozenset({ARTIFACT_METRICS, ARTIFACT_QUERY_TRACES})

# Provided externally by the dataset/runtime, so a node can require one without an
# upstream producer.
SOURCE_ARTIFACTS = frozenset(
    {
        ARTIFACT_QUERY_AUDIO,
        ARTIFACT_QUERY_TEXT,
        ARTIFACT_CORPUS,
        ARTIFACT_RELEVANT_DOCS,
        ARTIFACT_SHORT_ANSWERS,
    }
)

# A dataset_source's ``role`` param narrows what it advertises, so a graph can mix multiple
# sources and downstream nodes auto-wire to the right one by artifact name. ``both`` (default)
# advertises everything.
DATASET_ROLE_CORPUS = "corpus"
DATASET_ROLE_QUESTIONS = "questions"
DATASET_ROLE_BOTH = "both"

# Type-preserving T→T nodes that support the corpus axis (`axis: docs`).
# Extend as experiments need it (query_correction / query_optimization are candidates).
DOCS_AXIS_CAPABLE = frozenset({"augmenter"})

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
