"""Operator catalogue: the 11 ``register_stage_node`` declarations + their helpers.

This is the node catalogue proper — the field-parameterized operators (source / convert /
embed / combine / transform / index / search / refine / measure / generate / sink) and the
``callable(params)`` port/category/domain helpers that vary an operator's contract by field.

Importing this module runs the ``register_stage_node(...)`` calls (at import time, in the
order below), which populate ``registry._NODE_REGISTRY``. ``registry`` imports this module
at the bottom of its own file, after ``register_stage_node`` is defined, so the node
registration happens whenever ``registry`` is imported.
"""

from ...config.types import RETRIEVAL_MODES, RERANKER_MODES
from .artifacts import (
    ARTIFACT_ANSWER_SCORES,
    ARTIFACT_AUDIO_QUERY_VECTORS,
    ARTIFACT_AUGMENTED_QUERY_TEXT,
    ARTIFACT_CORPUS,
    ARTIFACT_CORPUS_VECTORS,
    ARTIFACT_CORRECTED_QUERY_TEXT,
    ARTIFACT_EMBEDDING_ALIGNMENT,
    ARTIFACT_FUSED_QUERY_VECTORS,
    ARTIFACT_GENERATED_ANSWERS,
    ARTIFACT_JUDGE_PASS,
    ARTIFACT_JUDGE_SCORES,
    ARTIFACT_METRICS,
    ARTIFACT_OPTIMIZED_QUERY_TEXT,
    ARTIFACT_QUERY_AUDIO,
    ARTIFACT_QUERY_TEXT,
    ARTIFACT_QUERY_TRACES,
    ARTIFACT_QUERY_VECTORS,
    ARTIFACT_REFERENCE_TRANSCRIPTION,
    ARTIFACT_REFINED_QUERY_TEXT,
    ARTIFACT_RELEVANT_DOCS,
    ARTIFACT_RETRIEVAL_SCORES,
    ARTIFACT_RETRIEVED,
    ARTIFACT_SHORT_ANSWERS,
    ARTIFACT_TEXT_QUERY_VECTORS,
    ARTIFACT_TRANSCRIPTION_SCORES,
    ARTIFACT_VECTOR_INDEX,
    QUERY_TEXT_CHAIN,
    one_of,
)
from .registry import register_stage_node

# Node catalogue. A node that runs a model declares the config field that selects it
# (required-fields validation + per-stage lifecycle derive from one place); inputs/outputs
# declare its data contract (edge validation). metrics has no required input — it appears
# in asr_only too and adapts to whatever ran.
#
# dataset_source is the graph root: it surfaces + validates the dataset's source
# artifacts so every downstream node hangs off it. (Dataset *loading* stays in prepare_dataset
# before the graph; TTS synthesis is in-graph — the tts node gap-fills any missing query audio.
# This node represents the loaded source in the executed graph.)
# ── operator: source (dataset_source / dataset_union) ────────────────
# dataset_source surfaces a dataset's source artifacts (role/fields narrowing lives in
# _effective_outputs, keyed by node_kind); `union: true` is dataset_union (merges every
# bound producer's question-axis sets). Resolved byte-identically.
_SOURCE_FULL = (
    ARTIFACT_QUERY_AUDIO,
    ARTIFACT_QUERY_TEXT,
    ARTIFACT_CORPUS,
    ARTIFACT_RELEVANT_DOCS,
    ARTIFACT_SHORT_ANSWERS,
    # Spoken-transcription GT (== question_text on current datasets): the sole producer
    # now that asr/audio_embedding are pure transforms (Phase 3).
    ARTIFACT_REFERENCE_TRANSCRIPTION,
)
_UNION_OUT = (
    ARTIFACT_QUERY_AUDIO,
    ARTIFACT_QUERY_TEXT,
    ARTIFACT_RELEVANT_DOCS,
    ARTIFACT_SHORT_ANSWERS,
)


def _source_outputs(params):
    return _UNION_OUT if (params or {}).get("union") else _SOURCE_FULL


def _source_optional(params):
    return _UNION_OUT if (params or {}).get("union") else ()


register_stage_node(
    "source",
    category="source",
    domain="ingest",
    inputs=(),
    outputs=_source_outputs,
    optional_inputs=_source_optional,
    param_spec={
        # Registered dataset id (picker in the builder) or a datasets-map id (B4).
        # A plain source just provides its outputs; the `role` narrowing (questions/corpus)
        # is an advanced multi-source mechanism set via the datasets map / node params, not a
        # default form field (to COMBINE datasets the builder offers the Dataset-union tile).
        "dataset": {"kind": "dataset"},
    },
)


# ── operator: convert (modality change) — asr (A→T) / tts (T→A) ──────
# asr is a managed MODEL (device/offload); tts runs its own TTS backend and is a
# transform — so `category`/`model_field` are field-dependent (callable). Resolved
# byte-identically to the two former nodes.
def _convert_is_tts(params):
    return (params or {}).get("op") == "tts"


register_stage_node(
    "convert",
    category=lambda p: "transform" if _convert_is_tts(p) else "model",
    domain=lambda p: "ingest" if _convert_is_tts(p) else "transcription",
    model_field=lambda p: None if _convert_is_tts(p) else "model.asr_model_type",
    inputs=lambda p: (ARTIFACT_QUERY_TEXT,) if _convert_is_tts(p) else (ARTIFACT_QUERY_AUDIO,),
    outputs=lambda p: (ARTIFACT_QUERY_AUDIO,) if _convert_is_tts(p) else (ARTIFACT_QUERY_TEXT,),
    param_spec={
        "op": {"kind": "select", "choices": ["asr", "tts"]},
        # oracle: the ASR branch uses the reference transcriptions instead of running ASR (R2).
        "oracle": {"kind": "bool", "default": False, "show_if": {"op": ["asr"]}},
    },
)


# ── operator: embed (collapses text/audio/corpus embedding) ───────────
# One model node whose contract varies by field (the §4.1 "axis insight" made literal):
#   axis=query, modality=text  → query_text chain → text_query_vectors   (model.text_emb)
#   axis=query, modality=audio → query_audio       → audio_query_vectors  (model.audio_emb)
#   axis=corpus                → corpus            → corpus_vectors       (shared embedder)
# Resolved byte-identically to the three former nodes (node_kind reverses each combo).
def _embed_inputs(params):
    p = params or {}
    if p.get("axis") == "corpus":
        return (ARTIFACT_CORPUS,)
    if p.get("modality") == "audio":
        return (ARTIFACT_QUERY_AUDIO,)
    return (QUERY_TEXT_CHAIN,)


def _embed_outputs(params):
    p = params or {}
    if p.get("axis") == "corpus":
        return (ARTIFACT_CORPUS_VECTORS,)
    if p.get("modality") == "audio":
        return (ARTIFACT_AUDIO_QUERY_VECTORS,)
    return (ARTIFACT_TEXT_QUERY_VECTORS,)


def _embed_model_field(params):
    p = params or {}
    if p.get("axis") == "corpus":
        return None  # corpus embedding shares the query embedder instance
    if p.get("modality") == "audio":
        return "model.audio_emb_model_type"
    return "model.text_emb_model_type"


register_stage_node(
    "embed",
    category="model",
    domain="embedding",
    model_field=_embed_model_field,
    inputs=_embed_inputs,
    outputs=_embed_outputs,
    param_spec={
        "axis": {"kind": "select", "choices": ["query", "corpus"], "default": "query"},
        "modality": {"kind": "select", "choices": ["text", "audio"], "default": "text",
                     "show_if": {"axis": ["query"]}},
    },
)


# ── operator: combine (collapses fusion / result_fusion / corpus_merge) ──
# Three combine LEVELS: embedding (fuse the audio+text query vectors → a distinct fused
# stream; text optional, falls back to audio), result (fuse two ranked `retrieved` sets),
# set (concatenate every bound corpus_vectors producer). Resolved byte-identically.
def _combine_inputs(params):
    level = (params or {}).get("level")
    if level == "result":
        return (ARTIFACT_RETRIEVED,)
    if level == "set":
        return (ARTIFACT_CORPUS_VECTORS,)
    return (ARTIFACT_AUDIO_QUERY_VECTORS,)  # embedding


def _combine_outputs(params):
    level = (params or {}).get("level")
    if level == "result":
        return (ARTIFACT_RETRIEVED,)
    if level == "set":
        return (ARTIFACT_CORPUS_VECTORS,)
    return (ARTIFACT_FUSED_QUERY_VECTORS,)  # embedding


def _combine_optional(params):
    return (
        (ARTIFACT_TEXT_QUERY_VECTORS,)
        if (params or {}).get("level", "embedding") == "embedding"
        else ()
    )


def _combine_domain(params):
    return "embedding" if (params or {}).get("level") == "set" else "fusion"


def _combine_param_spec(params):
    """Builder form for the combine node. The result-fusion `method` choices come from the fusion
    registry (`list_fusions`), so a newly-registered strategy appears in the UI automatically."""
    from ...models.retrieval import list_fusions

    return {
        "level": {"kind": "select", "choices": ["embedding", "result", "set"],
                  "default": "embedding"},
        # result-fusion params (the rank-fusion of two retrieved sets)
        "hybrid": {"kind": "bool", "default": False, "show_if": {"level": ["result"]}},
        "method": {"kind": "select", "choices": list_fusions(),
                   "default": "rrf", "show_if": {"level": ["result"]}},
        "weight": {"kind": "number", "default": 0.5, "show_if": {"level": ["result"]}},
        "k": {"kind": "number", "show_if": {"level": ["result"]}},
        "top_k": {"kind": "number", "show_if": {"level": ["result"]}},
        "rrf_k": {"kind": "number", "default": 60, "show_if": {"level": ["result"]}},
    }


register_stage_node(
    "combine",
    category="transform",
    domain=_combine_domain,
    inputs=_combine_inputs,
    outputs=_combine_outputs,
    optional_inputs=_combine_optional,
    param_spec=_combine_param_spec,
)


# Pre-retrieval LLM query optimization (rewrite / HyDE): a PURE text→text transform that
# improves the query before embedding (emits optimized_query_text on the chain). The fan-out
# methods (decompose / multi_query) live in the explicit multi_query_retrieval node.
# ── operator: transform (type-preserving X→X) ────────────────────────
# Collapses query_correction / query_optimization / query_refine / augmenter /
# augment_audio. `op` selects the rewrite; `axis` (query/docs) + `modality` (text/audio)
# flip the data axis (the §4.1 docs-axis robustness folds in here). Resolved byte-identically.
def _transform_inputs(params):
    p = params or {}
    op = p.get("op")
    if op == "refine":
        return (QUERY_TEXT_CHAIN, ARTIFACT_RETRIEVED)
    if op == "perturb":
        if p.get("modality") == "audio":
            return (ARTIFACT_QUERY_AUDIO,)
        if p.get("axis") == "docs":
            return (ARTIFACT_CORPUS,)
    return (QUERY_TEXT_CHAIN,)


def _transform_outputs(params):
    p = params or {}
    op = p.get("op")
    if op == "correct":
        return (ARTIFACT_CORRECTED_QUERY_TEXT,)
    if op == "optimize":
        return (ARTIFACT_OPTIMIZED_QUERY_TEXT,)
    if op == "refine":
        return (ARTIFACT_REFINED_QUERY_TEXT,)
    if p.get("modality") == "audio":  # perturb, audio axis
        return (ARTIFACT_QUERY_AUDIO,)
    if p.get("axis") == "docs":  # perturb, corpus axis
        return (ARTIFACT_CORPUS,)
    return (ARTIFACT_AUGMENTED_QUERY_TEXT,)  # perturb, query axis


def _transform_optional(params):
    p = params or {}
    # the query-side augmenter can flip to the corpus axis; the optional corpus orders it
    # after the corpus producer (matches the former augmenter's optional_inputs).
    if p.get("op") == "perturb" and p.get("modality") != "audio":
        return (ARTIFACT_CORPUS,)
    return ()


def _transform_domain(params):
    return "robustness" if (params or {}).get("op") == "perturb" else "query"


_OP_SELECTOR = {"kind": "select",
                "choices": ["correct", "optimize", "refine", "perturb"]}


def _transform_param_spec(params):
    """Field-aware builder form: the `op` picker, plus only the params of the chosen op (so
    the conflicting `method` key carries the *right* choices — registry correctors vs
    rewrite/HyDE vs the refine methods). Params-free (op unset) shows just the picker."""
    op = (params or {}).get("op")
    spec = {"op": _OP_SELECTOR}
    if op == "correct":
        from ...evaluation.query_correction import list_correctors  # registry-driven

        spec.update({
            "method": {"kind": "select", "choices": list_correctors(), "default": "rule"},
            "enabled": {"kind": "bool", "default": True},
            "use_default_rules": {"kind": "bool", "default": True},
            "kb_max_distance": {"kind": "number", "default": 1},
            "kb_terms": {"kind": "json"},
            "replacements": {"kind": "json"},
        })
    elif op == "optimize":
        spec.update({
            "method": {"kind": "select", "choices": ["rewrite", "hyde"]},
            "temperature": {"kind": "number"},
            "max_iterations": {"kind": "number"},
        })
    elif op == "refine":
        spec.update({
            "method": {"kind": "select",
                       "choices": ["rewrite_with_context", "relevance_feedback",
                                   "self_rag_critique"],
                       "default": "rewrite_with_context"},
            "context_top_k": {"kind": "number", "default": 3},
        })
    elif op == "perturb":
        spec.update({
            "axis": {"kind": "select", "choices": ["query", "docs"], "default": "query"},
            "modality": {"kind": "select", "choices": ["text", "audio"],
                         "default": "text"},
            "homophones": {"kind": "bool", "default": True},
            "unit_corruption": {"kind": "bool", "default": True},
            "char_swap_prob": {"kind": "number", "default": 0.0},
            "max_edits": {"kind": "number", "default": 2},
            "add_noise": {"kind": "bool", "default": True},
            "snr_db": {"kind": "number", "default": 20.0},
            "speed_perturbation": {"kind": "bool", "default": False},
            "pitch_shift": {"kind": "bool", "default": False},
            "volume_change": {"kind": "bool", "default": False},
            "n_variants": {"kind": "number", "default": 1},
        })
    return spec


register_stage_node(
    "transform",
    category="transform",
    domain=_transform_domain,
    inputs=_transform_inputs,
    outputs=_transform_outputs,
    optional_inputs=_transform_optional,
    param_spec=_transform_param_spec,
)
# Composite retrieval strategy (RAG-fusion) = the `search` operator with a `fanout`
# (multi_query / decompose): expand the query into sub-queries, retrieve each, fuse the
# result sets → retrieved. Runtime-variable fan-out; replaces embed+retrieve for that flow.
# query_refine / query_correction / augmenter = the `transform` operator with op:
# refine / correct / perturb (the query-side rewrites that chain on the query_text bus).
# Query-set union (§4.1 T3) = the `source` operator with union:true — merges every bound
# producer's question-axis ItemSets (refs/texts/GT) with DISJOINT query_ids enforced.
# Audio-axis robustness = the `transform` operator with op:perturb, modality:audio
# (perturbs each query clip — noise/speed/pitch/volume — and republishes query_audio REFS).
# Corpus side of the graph (§4 split): the corpus embedder is the `embed` operator with
# `axis: corpus` (same embedder instance the query side uses — offload planner accounts for
# both, via node_kind→corpus_embedding); vector_db owns the store backend choice per node.
# Corpus combiner = the `combine` operator with level:set (concatenate every bound
# corpus_vectors producer); the embedded corpora become one set in the DB.
# ── operator: index (was vector_db) — corpus_vectors → searchable index ──
register_stage_node(
    "index",
    category="transform",
    domain="retrieval",
    inputs=(ARTIFACT_CORPUS_VECTORS,),
    outputs=(ARTIFACT_VECTOR_INDEX,),
    # CSE: an explicit `store: inmemory` collapses with an omitted one (S7). (Carried over
    # from vector_db — the one node-type with a CSE-twin default.)
    param_defaults={"store": "inmemory"},
    param_spec={
        "store": {
            "kind": "select",
            "choices": ["inmemory", "faiss", "faiss_gpu", "chromadb", "qdrant"],
            "default": "inmemory",
        },
        "gpu_id": {"kind": "number", "show_if": {"store": ["faiss_gpu"]}},
        "path": {"kind": "text", "show_if": {"store": ["chromadb"]}},
        "url": {"kind": "text", "show_if": {"store": ["qdrant"]}},
        "collection": {"kind": "text", "show_if": {"store": ["chromadb", "qdrant"]}},
    },
)
# ── operator: search (collapses retrieval / multi_query_retrieval) ────
# Plain search: highest-priority published vector stream (fused > audio > text > a
# precomputed query_vectors column) + the index → ranked. A `fanout` (multi_query /
# decompose) REPLACES the vector input with the query text (the composite expands +
# retrieves per sub-query, runtime-variable). Resolved byte-identically.
_SEARCH_VECTOR_ONE_OF = one_of(
    ARTIFACT_FUSED_QUERY_VECTORS,
    ARTIFACT_AUDIO_QUERY_VECTORS,
    ARTIFACT_TEXT_QUERY_VECTORS,
    ARTIFACT_QUERY_VECTORS,
    key=ARTIFACT_QUERY_VECTORS,
)


def _is_fanout(params):
    return (params or {}).get("method") in ("multi_query", "decompose")


def _search_inputs(params):
    if _is_fanout(params):
        return (QUERY_TEXT_CHAIN, ARTIFACT_VECTOR_INDEX)
    return (_SEARCH_VECTOR_ONE_OF, ARTIFACT_VECTOR_INDEX)


def _search_optional(params):
    if _is_fanout(params):
        return ()
    return (QUERY_TEXT_CHAIN, ARTIFACT_REFERENCE_TRANSCRIPTION)


def _search_param_spec(params):
    """Builder form for the search node. The multi-query `combine_strategy` choices come from its
    registry (`list_combine_strategies`), so a newly-registered strategy appears in the UI."""
    from ...models.retrieval.query.optimization import list_combine_strategies

    return {
        "k": {"kind": "number", "default": 5},
        "mode": {"kind": "select", "choices": list(RETRIEVAL_MODES),
                 "default": "dense"},
        "distance": {"kind": "text"},
        "gpu_id": {"kind": "number"},
        # result-fusion: pin this instance to ONE vector stream (else the one_of default).
        "vectors": {
            "kind": "select",
            "choices": ["query_vectors", "audio_query_vectors", "text_query_vectors",
                        "fused_query_vectors"],
        },
        # fan-out composite (multi-query / decompose): replaces embed+retrieve.
        "method": {"kind": "select", "choices": ["multi_query", "decompose"]},
        "combine_strategy": {
            "kind": "select", "choices": list_combine_strategies(),
            "default": "rrf", "show_if": {"method": ["multi_query", "decompose"]}},
    }


register_stage_node(
    "search",
    category="transform",
    domain="retrieval",
    inputs=_search_inputs,
    outputs=(ARTIFACT_RETRIEVED,),
    optional_inputs=_search_optional,
    param_spec=_search_param_spec,
)
# Result-level fusion = the `combine` operator with level:result (fuse the RANKED results
# of two retrievals instead of their embeddings); consumes + produces `retrieved`.
# Post-retrieval refinement as three composable nodes (each consumes + produces retrieved):
# rerank (reorder) -> mmr (diversity) -> threshold (filter). Each is present only when
# configured; they chain in that order and the last feeds metrics/answer_gen.
_QUERY_VECTOR_ONE_OF = one_of(
    ARTIFACT_FUSED_QUERY_VECTORS,
    ARTIFACT_AUDIO_QUERY_VECTORS,
    ARTIFACT_TEXT_QUERY_VECTORS,
    ARTIFACT_QUERY_VECTORS,
    key=ARTIFACT_QUERY_VECTORS,
)


# ── operator: refine (collapses rerank / mmr / threshold) ─────────────
# Each is retrieved→retrieved; only the optional scoring inputs differ by op. Resolved
# byte-identically to the three former nodes.
def _refine_optional(params):
    op = (params or {}).get("op")
    if op == "rerank":
        return (_QUERY_VECTOR_ONE_OF, QUERY_TEXT_CHAIN, ARTIFACT_REFERENCE_TRANSCRIPTION)
    if op == "mmr":
        return (_QUERY_VECTOR_ONE_OF,)
    return ()  # threshold


register_stage_node(
    "refine",
    category="transform",
    domain="refine",
    inputs=(ARTIFACT_RETRIEVED,),
    outputs=(ARTIFACT_RETRIEVED,),
    optional_inputs=_refine_optional,
    param_spec={
        "op": {"kind": "select", "choices": ["rerank", "mmr", "threshold"]},
        "mode": {"kind": "select",
                 "choices": list(RERANKER_MODES),
                 "show_if": {"op": ["rerank"]}},
        # per-node target depth (shared by the refine chain); rerank keeps the larger
        # fetch_k pool when an mmr node follows.
        "k": {"kind": "number"},
        "top_k": {"kind": "number", "show_if": {"op": ["rerank"]}},
    },
)
# metrics/answer_gen/finalize use optional_inputs to encode ordering: metrics after
# retrieval (when retrieving) and after asr (asr_only); answer_gen after metrics (it reads
# metrics intermediates); finalize last (after answer_gen when present).
# Typed comparison nodes (Phase 5): each pairs an EXPECTED (dataset_source GT) against an
# ACTUAL (a transform output) and is present only when both exist (diagram honesty).
# They set the per-item RunState intermediates the report + rag/judge stages read, and
# publish a scores artifact the `metrics` node orders after.
# ── operator: measure (typed comparisons + report + traces + judge) ──
# Collapses transcription/retrieval/answer/embedding_alignment metrics, the `metrics`
# report assembler, `build_query_traces`, and `answer_judge`. `family` (or `trace: true`)
# selects which; each pairs an EXPECTED GT × ACTUAL output → a scores artifact (the report
# orders after them). build_query_traces is the one transform-category member (callable
# category). Resolved byte-identically to the seven former nodes.
_MEASURE = {
    "transcription": {"inputs": (), "outputs": (ARTIFACT_TRANSCRIPTION_SCORES,),
                      "optional": (ARTIFACT_QUERY_TEXT, ARTIFACT_REFERENCE_TRANSCRIPTION)},
    "retrieval": {"inputs": (), "outputs": (ARTIFACT_RETRIEVAL_SCORES,),
                  "optional": (ARTIFACT_RETRIEVED, ARTIFACT_RELEVANT_DOCS)},
    "alignment": {"inputs": (ARTIFACT_AUDIO_QUERY_VECTORS, ARTIFACT_TEXT_QUERY_VECTORS),
                  "outputs": (ARTIFACT_EMBEDDING_ALIGNMENT,), "optional": ()},
    "answer": {"inputs": (ARTIFACT_GENERATED_ANSWERS,), "outputs": (ARTIFACT_ANSWER_SCORES,),
               "optional": (ARTIFACT_RETRIEVED, ARTIFACT_RELEVANT_DOCS)},
    "judge": {"inputs": (ARTIFACT_METRICS,),
              "outputs": (ARTIFACT_JUDGE_SCORES, ARTIFACT_JUDGE_PASS),
              "optional": (ARTIFACT_QUERY_TRACES, ARTIFACT_GENERATED_ANSWERS,
                           ARTIFACT_RETRIEVED)},
    "trace": {"inputs": (), "outputs": (ARTIFACT_QUERY_TRACES,),
              "optional": (ARTIFACT_RETRIEVED, ARTIFACT_GENERATED_ANSWERS,
                           ARTIFACT_METRICS, ARTIFACT_REFERENCE_TRANSCRIPTION)},
    "report": {"inputs": (), "outputs": (ARTIFACT_METRICS,),
               "optional": (ARTIFACT_TRANSCRIPTION_SCORES, ARTIFACT_RETRIEVAL_SCORES,
                            ARTIFACT_RETRIEVED, ARTIFACT_QUERY_TEXT,
                            ARTIFACT_REFERENCE_TRANSCRIPTION, ARTIFACT_RELEVANT_DOCS,
                            ARTIFACT_EMBEDDING_ALIGNMENT)},
}


def _measure_kind(params):
    p = params or {}
    return "trace" if p.get("trace") else p.get("family", "report")


register_stage_node(
    "measure",
    category=lambda p: "transform" if _measure_kind(p) == "trace" else "metric",
    domain=lambda p: "reporting" if _measure_kind(p) == "trace" else "scoring",
    inputs=lambda p: _MEASURE[_measure_kind(p)]["inputs"],
    outputs=lambda p: _MEASURE[_measure_kind(p)]["outputs"],
    optional_inputs=lambda p: _MEASURE[_measure_kind(p)]["optional"],
    param_spec={
        "family": {"kind": "select",
                   "choices": ["transcription", "retrieval", "answer", "alignment",
                               "judge", "report"]},
        "trace": {"kind": "bool", "default": False},
    },
)
# ── operator: generate (was answer_gen) — query (+ context?) → answers ──
register_stage_node(
    "generate",
    category="transform",
    domain="generation",
    # The query is the only hard requirement: `retrieved` is OPTIONAL *context* — present →
    # RAG (grounded in the retrieved docs), absent → closed-book QA (a no-corpus dataset has
    # no retrieval node, so no `retrieved` artifact). The effective (most-processed) query
    # text drives generation either way; in retrieval modes the optional `retrieved` still
    # orders answer_gen after retrieval via its producer binding (parity-preserving).
    inputs=(QUERY_TEXT_CHAIN,),
    outputs=(ARTIFACT_GENERATED_ANSWERS,),
    optional_inputs=(
        ARTIFACT_RETRIEVED,
        ARTIFACT_METRICS,
        ARTIFACT_REFERENCE_TRANSCRIPTION,
    ),
)
# answer_metrics / build_query_traces / answer_judge = the `measure` operator with
# family:answer / trace:true / family:judge (see the measure registration above).
# ── operator: sink (terminal side-effects) ───────────────────────────
# Collapses finalize / aggregate / dataset_sink / leaderboard_sink / tracking_sink — the
# `target` field selects which terminal effect (+ its optional ordering inputs).
_SINK_OPTIONAL = {
    "finalize": (ARTIFACT_QUERY_TRACES, ARTIFACT_JUDGE_SCORES,
                 ARTIFACT_GENERATED_ANSWERS, ARTIFACT_RETRIEVED),
    "aggregate": (ARTIFACT_RETRIEVED, ARTIFACT_METRICS),
    "dataset": (ARTIFACT_QUERY_AUDIO, ARTIFACT_GENERATED_ANSWERS),
    "leaderboard": (ARTIFACT_METRICS,),
    "tracking": (ARTIFACT_METRICS,),
}


def _sink_inputs(params):
    # finalize is the only sink with a required input (the report); the rest are pure
    # optional-ordered terminals.
    return (ARTIFACT_METRICS,) if (params or {}).get("target") == "finalize" else ()


def _sink_optional(params):
    return _SINK_OPTIONAL.get((params or {}).get("target"), ())


def _sink_domain(params):
    return "reporting" if (params or {}).get("target") in ("finalize", "aggregate") \
        else "export"


register_stage_node(
    "sink",
    category="sink",
    domain=_sink_domain,
    inputs=_sink_inputs,
    outputs=(),
    optional_inputs=_sink_optional,
    param_spec={
        "target": {"kind": "select",
                   "choices": ["finalize", "aggregate", "dataset", "leaderboard",
                               "tracking"]},
    },
)
