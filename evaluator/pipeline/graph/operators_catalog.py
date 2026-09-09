"""Operator catalogue: the ``register_stage_node`` declarations and their field-parameterized
port/category/domain helpers. Importing this module populates ``registry.NODE_REGISTRY``;
``registry`` imports it at the bottom of its own file, after ``register_stage_node`` is defined."""

from ...config.types import RETRIEVAL_MODES, RERANKER_MODES
from ...models.retrieval import list_fusions
from ...models.retrieval.query.correction import list_correctors
from ...models.retrieval.query.optimization import list_combine_strategies
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
    ARTIFACT_REFERENCE_TEXT,
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

# source (dataset_source) is the graph root, surfacing the dataset's source artifacts; dataset
# loading stays in prepare_dataset before the graph, TTS synthesis is in-graph. `union: true` is
# dataset_union: merges every bound producer's question-axis sets; role/fields narrowing lives
# in effective_outputs, keyed by node_kind.
_SOURCE_FULL = (
    ARTIFACT_QUERY_AUDIO,
    ARTIFACT_QUERY_TEXT,
    ARTIFACT_CORPUS,
    ARTIFACT_RELEVANT_DOCS,
    ARTIFACT_SHORT_ANSWERS,
    # Spoken-transcription GT (== question_text on current datasets); asr/audio_embedding
    # are pure transforms, not its producer.
    ARTIFACT_REFERENCE_TRANSCRIPTION,
)
# Appended (not inlined above) so the tuple-order-derived UI column order stays untouched.
_SOURCE_FULL = _SOURCE_FULL + (ARTIFACT_REFERENCE_TEXT,)
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
        # `role` narrowing (questions/corpus) is a multi-source mechanism set via the datasets
        # map, not a form field — the builder offers the Dataset-union tile to combine datasets.
        "dataset": {"kind": "dataset"},
    },
)


# asr (managed model, device/offload) and tts (own backend, a transform) are different node
# types rather than one field-resolved contract.
register_stage_node(
    "asr",
    category="model",
    domain="transcription",
    model_field="model.asr_model_type",
    inputs=(ARTIFACT_QUERY_AUDIO,),
    outputs=(ARTIFACT_QUERY_TEXT,),
    # Optional input: its scoring ground truth (query_text is scored against it by WER/CER).
    optional_inputs=(ARTIFACT_REFERENCE_TRANSCRIPTION,),
    param_spec={
        # oracle: uses the reference transcriptions instead of running ASR.
        "oracle": {"kind": "bool", "default": False},
    },
)

register_stage_node(
    "tts",
    category="transform",
    domain="ingest",
    inputs=(ARTIFACT_QUERY_TEXT,),
    outputs=(ARTIFACT_QUERY_AUDIO,),
    param_spec={
        # provider mirrors the @register_tts_model registry; add a name when one is registered.
        "provider": {"kind": "select", "choices": ["piper", "mms", "m4t", "xtts_v2"],
                     "default": "piper"},
        "voice": {"kind": "text", "default": "en_US-lessac-medium"},
        "language": {"kind": "text", "default": "en"},
        "sample_rate": {"kind": "number", "default": 16000},
        "speed": {"kind": "number", "default": 1.0},
        "pitch": {"kind": "number", "default": 1.0},
        "volume": {"kind": "number", "default": 1.0},
        "seed": {"kind": "number", "default": 42},
        "output_dir": {"kind": "text", "default": "prepared_benchmarks/audio"},
    },
)


# One model node whose contract varies by field: axis=corpus->corpus_vectors (shared embedder);
# axis=query,modality=audio->audio_query_vectors; else query_text chain->text_query_vectors.
# `node_kind` reverses each combo.
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
        return None  # shares the query embedder instance
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


# 3 levels: embedding (fuse audio+text query vectors, text optional/falls back to audio),
# result (fuse two ranked `retrieved` sets), set (concatenate bound corpus_vectors producers).
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

    return {
        "level": {"kind": "select", "choices": ["embedding", "result", "set"],
                  "default": "embedding"},
        "audio_weight": {"kind": "number", "default": 0.5, "show_if": {"level": ["embedding"]}},
        "text_weight": {"kind": "number", "default": 0.5, "show_if": {"level": ["embedding"]}},
        "fusion_method": {"kind": "select",
                          "choices": ["weighted", "concatenate", "max_pool", "average"],
                          "default": "weighted", "show_if": {"level": ["embedding"]}},
        "normalize_before_fusion": {"kind": "bool", "default": True,
                                    "show_if": {"level": ["embedding"]}},
        "require_same_dimensions": {"kind": "bool", "default": False,
                                    "show_if": {"level": ["embedding"]}},
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


# `op` selects the rewrite; `axis` (query/docs) + `modality` (text/audio) flip the data axis.
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
    # after the corpus producer.
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
        spec.update({
            "method": {"kind": "select", "choices": list_correctors(), "default": "rule"},
            "enabled": {"kind": "bool", "default": True},
            "use_default_rules": {"kind": "bool", "default": True},
            "kb_max_distance": {"kind": "number", "default": 1},
            "phonetic_max_edits": {"kind": "number", "default": 2},
            "constrain_to_vocab": {"kind": "bool", "default": False},
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
# `union:true` (on `transform`) merges bound producers' question-axis ItemSets with disjoint
# query_ids enforced; op:perturb,modality:audio republishes perturbed query_audio refs.
register_stage_node(
    "index",
    category="transform",
    domain="retrieval",
    inputs=(ARTIFACT_CORPUS_VECTORS,),
    outputs=(ARTIFACT_VECTOR_INDEX,),
    # No `param_defaults` for `store`: an omitted value inherits the run's global vector_db.type,
    # NOT equivalent to an explicit `inmemory` default (would over-share under a non-inmemory
    # global).
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
# Vector input is a OneOf, priority fused > audio > text > precomputed query_vectors; a
# `fanout` (multi_query/decompose) replaces it with the query text instead.
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
    # query_text/reference_transcription are functional (sparse/hybrid + audio_text scoring);
    # relevant_docs is retrieval's ground truth, consumed by the derived retrieval_metrics node.
    return (QUERY_TEXT_CHAIN, ARTIFACT_REFERENCE_TRANSCRIPTION, ARTIFACT_RELEVANT_DOCS)


def _search_param_spec(params):
    """Builder form for the search node. The multi-query `combine_strategy` choices come from its
    registry (`list_combine_strategies`), so a newly-registered strategy appears in the UI."""

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
# Post-retrieval refinement, 3 composable retrieved->retrieved nodes: rerank (reorder) ->
# mmr (diversity) -> threshold (filter), each present only when configured.
_QUERY_VECTOR_ONE_OF = one_of(
    ARTIFACT_FUSED_QUERY_VECTORS,
    ARTIFACT_AUDIO_QUERY_VECTORS,
    ARTIFACT_TEXT_QUERY_VECTORS,
    ARTIFACT_QUERY_VECTORS,
    key=ARTIFACT_QUERY_VECTORS,
)


# Only the optional scoring inputs differ by op.
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
        # k: shared by the refine chain; rerank keeps the larger fetch_k pool when mmr follows.
        "k": {"kind": "number"},
        "top_k": {"kind": "number", "show_if": {"op": ["rerank"]}},
        # weight/device: rerank-only tuning; mmr's lambda / threshold's cutoff are global
        # config, not node-settable, so intentionally absent here.
        "weight": {"kind": "number", "show_if": {"op": ["rerank"]}},
        "device": {"kind": "device", "show_if": {"op": ["rerank"]}},
    },
)
# metrics/answer_gen/finalize use optional_inputs to encode ordering (metrics after
# retrieval/asr, answer_gen after metrics, finalize last). Typed comparison nodes pair an
# EXPECTED (dataset_source GT) against an ACTUAL (a transform output) and appear only when
# both exist; each publishes a scores artifact the `metrics` node orders after.
# Per-aspect judge outputs: a finite set, publish is config-conditional, advertising is not.
from ...config.judge import VALID_JUDGE_ASPECTS as _VALID_JUDGE_ASPECTS  # noqa: E402

_JUDGE_ASPECT_OUT = tuple(
    f"judge_aspect_{a}" for a in sorted(_VALID_JUDGE_ASPECTS)
)
# `family` (or `trace: true`) selects which measure kind runs; build_query_traces ("trace") is
# the one transform-category member, the rest are metrics comparing EXPECTED GT x ACTUAL output.
_MEASURE = {
    "transcription": {"inputs": (),
                      "outputs": (ARTIFACT_TRANSCRIPTION_SCORES,
                                  "per_query_wer", "per_query_cer"),
                      "optional": (ARTIFACT_QUERY_TEXT, ARTIFACT_REFERENCE_TRANSCRIPTION)},
    "retrieval": {"inputs": (),
                  "outputs": (ARTIFACT_RETRIEVAL_SCORES, "per_query_recall5"),
                  "optional": (ARTIFACT_RETRIEVED, ARTIFACT_RELEVANT_DOCS)},
    "alignment": {"inputs": (ARTIFACT_AUDIO_QUERY_VECTORS, ARTIFACT_TEXT_QUERY_VECTORS),
                  "outputs": (ARTIFACT_EMBEDDING_ALIGNMENT,), "optional": ()},
    "answer": {"inputs": (ARTIFACT_GENERATED_ANSWERS,), "outputs": (ARTIFACT_ANSWER_SCORES,),
               # short_answers: dataset's yes/no/maybe label, exact-matched against the
               # reply's `Short Answer:` line.
               "optional": (ARTIFACT_RETRIEVED, ARTIFACT_RELEVANT_DOCS,
                            ARTIFACT_SHORT_ANSWERS)},
    "judge": {"inputs": (ARTIFACT_QUERY_TRACES,),
              # query_traces rides the artifact bus, not `s.results`, so it survives the
              # report node's rebuild regardless of order — no ordering edge to `metrics` needed.
              # Every aspect artifact is advertised so finalize can bind it even when unpublished.
              "outputs": (ARTIFACT_JUDGE_SCORES, ARTIFACT_JUDGE_PASS) + _JUDGE_ASPECT_OUT,
              "optional": (ARTIFACT_GENERATED_ANSWERS,
                           ARTIFACT_RETRIEVED, ARTIFACT_RELEVANT_DOCS,
                           "per_query_recall5")},
    "trace": {"inputs": (), "outputs": (ARTIFACT_QUERY_TRACES,),
              "optional": (ARTIFACT_RETRIEVED, ARTIFACT_GENERATED_ANSWERS,
                           ARTIFACT_REFERENCE_TRANSCRIPTION,
                           ARTIFACT_RELEVANT_DOCS,
                           "per_query_wer", "per_query_cer", "per_query_recall5")},
    "report": {"inputs": (), "outputs": (ARTIFACT_METRICS,),
               "optional": (ARTIFACT_TRANSCRIPTION_SCORES, ARTIFACT_RETRIEVAL_SCORES,
                            ARTIFACT_ANSWER_SCORES,
                            ARTIFACT_RETRIEVED, ARTIFACT_QUERY_TEXT,
                            ARTIFACT_REFERENCE_TRANSCRIPTION, ARTIFACT_RELEVANT_DOCS,
                            ARTIFACT_EMBEDDING_ALIGNMENT,
                            # the corrected metrics read it
                            ARTIFACT_CORRECTED_QUERY_TEXT,
                            # per-item scores for the IR diagnostics
                            "per_query_wer", "per_query_cer", "per_query_recall5")},
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
        "enabled": {"kind": "bool", "default": False, "show_if": {"family": ["judge"]}},
        "model": {"kind": "text", "default": "gpt-4o-mini", "show_if": {"family": ["judge"]}},
        "judge_mode": {"kind": "select", "choices": ["retrieval", "answer_quality", "both"],
                       "default": "both", "show_if": {"family": ["judge"]}},
        "judge_aspects": {"kind": "json", "show_if": {"family": ["judge"]}},
        "score_aggregation": {"kind": "select", "choices": ["average", "weighted", "min", "max"],
                              "default": "average", "show_if": {"family": ["judge"]}},
        "aspect_weights": {"kind": "json", "show_if": {"family": ["judge"]}},
        "reference_mode": {"kind": "select", "choices": ["free", "graded"],
                           "default": "free", "show_if": {"family": ["judge"]}},
        "pass_threshold": {"kind": "number", "default": 0.5, "show_if": {"family": ["judge"]}},
        "max_cases": {"kind": "number", "default": 50, "show_if": {"family": ["judge"]}},
        "temperature": {"kind": "number", "default": 0.0, "show_if": {"family": ["judge"]}},
        "timeout_s": {"kind": "number", "default": 60, "show_if": {"family": ["judge"]}},
        "include_doc_text": {"kind": "bool", "default": True, "show_if": {"family": ["judge"]}},
        "judge_top_k": {"kind": "number", "default": 5, "show_if": {"family": ["judge"]}},
        "use_local_server": {"kind": "bool", "default": False, "show_if": {"family": ["judge"]}},
        "local_server_url": {"kind": "text", "show_if": {"family": ["judge"]}},
        "api_base": {"kind": "text", "default": "https://api.openai.com/v1/chat/completions",
                     "show_if": {"family": ["judge"]}},
        "api_key_env": {"kind": "text", "default": "OPENAI_API_KEY",
                        "show_if": {"family": ["judge"]}},
        "system_prompt": {"kind": "text", "show_if": {"family": ["judge"]}},
        "user_prompt_template": {"kind": "text", "show_if": {"family": ["judge"]}},
    },
)
register_stage_node(
    "generate",
    category="transform",
    domain="generation",
    # `retrieved` is optional context: present -> RAG, absent -> closed-book QA; also orders
    # answer_gen after retrieval. `short_answers` is the answer's ground truth (scored by
    # the derived answer_metrics node), wired from the source.
    inputs=(QUERY_TEXT_CHAIN,),
    outputs=(ARTIFACT_GENERATED_ANSWERS,),
    optional_inputs=(ARTIFACT_RETRIEVED, ARTIFACT_SHORT_ANSWERS, ARTIFACT_RELEVANT_DOCS),
    param_spec={
        "enabled": {"kind": "bool", "default": False},
        "method": {"kind": "select", "choices": ["simple", "chain_of_thought", "multi_query"],
                   "default": "simple"},
        "model": {"kind": "text", "default": "gpt-4o-mini"},
        "temperature": {"kind": "number", "default": 0.0},
        "context_docs": {"kind": "number", "default": 3},
        "context_max_chars": {"kind": "number", "default": 600},
        "max_cases": {"kind": "number", "default": 0},
        "timeout_s": {"kind": "number", "default": 120},
        "compute_rouge": {"kind": "bool", "default": True},
        "use_local_server": {"kind": "bool", "default": False},
        "local_server_url": {"kind": "text"},
        "api_base": {"kind": "text", "default": "https://api.openai.com/v1/chat/completions"},
        "api_key_env": {"kind": "text", "default": "OPENAI_API_KEY"},
        "system_prompt": {"kind": "text"},
        "prompt_template": {"kind": "text"},
        "reference_metadata_field": {"kind": "text", "default": "long_answer"},
    },
)
# answer_metrics/build_query_traces/answer_judge = `measure` with family:answer/trace:true/
# family:judge. `target` selects the sink's terminal effect + its optional ordering inputs.
_SINK_OPTIONAL = {
    # judge_pass + the aspect artifacts must be declared here, or streaming's lifetime
    # analysis misses the finalize sink's reads.
    "finalize": (ARTIFACT_QUERY_TRACES, ARTIFACT_JUDGE_SCORES, ARTIFACT_JUDGE_PASS,
                 ARTIFACT_GENERATED_ANSWERS, ARTIFACT_RETRIEVED) + _JUDGE_ASPECT_OUT,
    # everything the branched report reads, declared so the reads are visible in the graph
    "aggregate": (ARTIFACT_RETRIEVED, ARTIFACT_METRICS, ARTIFACT_QUERY_TEXT,
                  ARTIFACT_CORRECTED_QUERY_TEXT, ARTIFACT_RELEVANT_DOCS,
                  ARTIFACT_REFERENCE_TEXT, ARTIFACT_REFERENCE_TRANSCRIPTION),
    "dataset": (ARTIFACT_QUERY_AUDIO, ARTIFACT_GENERATED_ANSWERS),
    "leaderboard": (ARTIFACT_METRICS,),
    "tracking": (ARTIFACT_METRICS,),
}


def _sink_inputs(params):
    # finalize is the only sink with a required input; the rest are pure optional-ordered terminals.
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
