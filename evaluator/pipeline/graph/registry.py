"""Node-type + stage-graph registry: the DAG's structural vocabulary.

``register_stage_node`` declares a node *type* (its name + the config field of the model
it runs, if any). This is the structural twin of the executable handler registry in
``evaluation/stage_registry.py``: a node declares itself once and introspection (model
lifecycle, required-field derivation) reads it, instead of a hardcoded per-mode map.

Also hosts the artifact vocabulary, the ``StageNode`` / ``StageGraph`` data structures,
and ``validate_graph_artifacts`` (satisfiability check over topological order).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

from .taxonomy import validate_taxonomy


@dataclass(frozen=True)
class StageNode:
    """Single stage node in the execution DAG.

    ``bindings`` resolve each consumed artifact to its producer node id(s):
    ``((artifact_name, producer_id), ...)`` in producer order. With one producer the
    consumer reads that node's output; with several (e.g. fusion's two embedders) all are
    listed. The RunContext executor (R3) reads inputs via these bindings. ``depends_on``
    (ordering) is the set of producer ids and stays the contract for topological sort.
    """

    id: str
    stage: str
    depends_on: Tuple[str, ...] = ()
    bindings: Tuple[Tuple[str, str], ...] = ()
    # Canonical-key → ordered candidate artifact names for OneOf inputs (e.g.
    # ("query_text", ("optimized_query_text", "corrected_query_text", "query_text"))).
    # Empty for plain inputs. ``s.input(key)`` reads the highest-priority candidate a
    # bound producer actually published.
    input_aliases: Tuple[Tuple[str, Tuple[str, ...]], ...] = ()
    # Optional per-input *role* tags ((artifact, producer_id, role)), parallel to ``bindings``
    # (operator-abstraction): lets a generic node interpret an input by role —
    # comparison-by-edge for ``measure`` (expected/actual). Empty for every node that doesn't
    # opt in, so it is excluded from the CSE key (cse.py reads only ``bindings``) and existing
    # graphs are byte-identical.
    binding_roles: Tuple[Tuple[str, str, str], ...] = ()
    # Per-instance config (duplicate/arbitrary nodes); None for single-instance modes.
    params: Optional[dict] = field(default=None, compare=False, hash=False)


@dataclass(frozen=True)
class StageGraph:
    """Execution graph with deterministic topological levels."""

    mode: str
    nodes: Tuple[StageNode, ...]

    def node_ids(self) -> List[str]:
        return [node.id for node in self.nodes]

    def has_stage(self, stage: str) -> bool:
        return any(node.stage == stage for node in self.nodes)

    def topological_levels(self) -> List[List[StageNode]]:
        remaining: Dict[str, StageNode] = {node.id: node for node in self.nodes}
        resolved: set[str] = set()
        levels: List[List[StageNode]] = []

        while remaining:
            ready = sorted(
                [
                    node
                    for node in remaining.values()
                    if all(dep in resolved for dep in node.depends_on)
                ],
                key=lambda node: node.id,
            )
            if not ready:
                unresolved = ", ".join(sorted(remaining.keys()))
                raise ValueError(f"Cycle detected in stage graph: {unresolved}")

            levels.append(ready)
            for node in ready:
                resolved.add(node.id)
                remaining.pop(node.id, None)

        return levels


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
ARTIFACT_JUDGE_SCORES = "judge_scores"
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


def _resolve(field: Any, params: Optional[dict]) -> Any:
    """Resolve a value-or-callable ``StageNodeDef`` field (operator-abstraction).

    A node-type's ports / category / domain / model_field may be a static value OR a
    ``callable(params)`` — the latter lets one generic operator vary its contract by field
    (e.g. ``embed`` produces ``corpus_vectors`` vs ``text_query_vectors`` by ``axis``). For
    the pre-collapse static registrations this is a no-op identity."""
    return field(params or {}) if callable(field) else field


def _effective_outputs(stage: str, params: Optional[dict]) -> Tuple[str, ...]:
    """Outputs a node instance advertises, honoring dataset_source ``role`` + ``fields``.

    Non-``dataset_source`` nodes always advertise their registered outputs. A
    ``dataset_source`` with a ``fields`` param ({column: artifact}, injected from the
    dataset's descriptor) advertises exactly those artifacts; ``role:
    corpus``/``questions`` then narrows the slice; ``both`` (default) keeps the set."""
    ndef = get_stage_node_def(stage)
    # Corpus-axis transform instance (§4.1 T2): `axis: docs` flips a type-preserving
    # T→T node to consume + produce the corpus instead of query_text.
    if stage in _DOCS_AXIS_CAPABLE and (params or {}).get("axis") == "docs":
        return (ARTIFACT_CORPUS,)
    from .operators import node_kind  # lazy: avoid an import cycle

    # The dataset_source role/fields narrowing applies whether authored as the legacy name
    # or the `source` operator (node_kind reverses union:false → dataset_source).
    if node_kind(stage, params) != "dataset_source":
        return _resolve(ndef.outputs, params)
    fields = (params or {}).get("fields") or {}
    _full: Tuple[str, ...] = _resolve(ndef.outputs, params)  # the source's full output set
    base: Tuple[str, ...] = _full
    if fields:
        # Declared column schema: dedupe artifact names, keep registered-output order
        # first (stable wiring), then any extra registered-elsewhere artifacts.
        declared = list(dict.fromkeys(fields.values()))
        base = tuple(
            [a for a in _full if a in declared]
            + [a for a in declared if a not in _full]
        )
    role = (params or {}).get("role") or DATASET_ROLE_BOTH
    if role == DATASET_ROLE_BOTH:
        return base
    if role not in _DATASET_SOURCE_ROLE_OUTPUTS:
        allowed = sorted([*_DATASET_SOURCE_ROLE_OUTPUTS, DATASET_ROLE_BOTH])
        raise ValueError(f"Unknown dataset_source role '{role}'. Allowed: {allowed}")
    narrowed = _DATASET_SOURCE_ROLE_OUTPUTS[role]
    if fields:
        # Fields narrow within the role slice; an empty intersection means the
        # schema doesn't describe this usage (e.g. a corpus-role source resolved
        # to a transcription dataset's defaults) — the role stays authoritative.
        inter = tuple(a for a in base if a in narrowed)
        return inter if inter else narrowed
    return narrowed


def _effective_inputs(stage: str, params: Optional[dict]) -> Tuple[str, ...]:
    """Required inputs a node instance consumes (display + wiring truthfulness).

    A docs-axis transform (§4.1 T2) consumes the corpus, not query_text."""
    ndef = get_stage_node_def(stage)
    if stage in _DOCS_AXIS_CAPABLE and (params or {}).get("axis") == "docs":
        return (ARTIFACT_CORPUS,)
    return display_artifact_names(_resolve(ndef.inputs, params))


def dataset_columns(params: Optional[dict]) -> List[Dict[str, str]]:
    """The display schema of a dataset_source node instance: ``[{name, artifact, type}]``.

    Built from the injected ``params.fields`` column schema (+ the artifact registry's
    modality as the human-readable type). Empty when the node carries no schema."""
    from ..artifacts import artifact_modality, is_registered

    fields = (params or {}).get("fields") or {}
    columns: List[Dict[str, str]] = []
    for name, artifact in fields.items():
        columns.append(
            {
                "name": str(name),
                "artifact": str(artifact),
                "type": (
                    str(artifact_modality(str(artifact)).value)
                    if is_registered(str(artifact))
                    else "?"
                ),
            }
        )
    return columns


@dataclass(frozen=True)
class StageNodeDef:
    """Declared node *type*: name, the config field of the model it runs, and the
    artifacts it consumes (``inputs``) / produces (``outputs``)."""

    stage: str
    # Declared class (taxonomy.py): `category` is the data-flow role (source/model/transform/
    # metric/sink) — load-bearing for lifecycle/validation/execution; `domain` is the
    # functional area (UI grouping / DAG coloring / docs). Both required at registration.
    category: str = "transform"
    domain: str = "ingest"
    model_field: Optional[str] = None  # e.g. "model.asr_model_type"; None = model-free
    inputs: Tuple[str, ...] = ()  # required artifacts (validated; gate execution)
    outputs: Tuple[str, ...] = ()  # artifacts this node makes available
    # Optional inputs express ordering / data that may be absent (e.g. metrics runs after
    # retrieval *when* retrieval is present). Auto-wiring adds a dependency on their
    # producer only when some node in the graph produces them; never a hard requirement.
    optional_inputs: Tuple[str, ...] = ()
    # Declared param defaults (S7): a param explicitly set to its default is canonically
    # identical to omitting it, so CSE collapses an explicit-vs-default twin into one run.
    param_defaults: Mapping[str, Any] = field(default_factory=dict)
    # Node-scoped builder params (NOT model params — those come from the chosen model's
    # registry `Params` schema): {key: {kind: bool|number|select|text|json, choices?,
    # default?}}. Only genuinely node-level switches belong here (e.g. asr.oracle).
    param_spec: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)


_NODE_REGISTRY: Dict[str, StageNodeDef] = {}


def register_stage_node(
    stage: str,
    *,
    category: str,
    domain: str,
    model_field: Optional[str] = None,
    inputs: Tuple[str, ...] = (),
    outputs: Tuple[str, ...] = (),
    optional_inputs: Tuple[str, ...] = (),
    param_defaults: Optional[Mapping[str, Any]] = None,
    param_spec: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> StageNodeDef:
    """Register a node type. ``category``/``domain`` are its declared class (taxonomy.py:
    role + functional area, both required + validated). ``model_field`` names the config key
    for its model; ``inputs``/``outputs`` are the required-consumed/produced artifacts;
    ``optional_inputs`` are wired to their producer only when present (ordering / optional
    data). ``param_defaults`` lets CSE treat an explicit-default param as an omitted one (S7).
    ``param_spec`` declares the node-scoped builder switches (model params come from the model
    registry instead)."""
    if stage in _NODE_REGISTRY:
        raise ValueError(f"Stage node already registered: {stage}")
    validate_taxonomy(stage, category, domain)
    node = StageNodeDef(
        stage=stage,
        category=category,
        domain=domain,
        model_field=model_field,
        inputs=inputs,
        outputs=outputs,
        optional_inputs=optional_inputs,
        param_defaults=dict(param_defaults or {}),
        # param_spec may be a callable(params) for an operator whose builder form is
        # field-aware (e.g. transform's `method` choices differ by `op`) — resolved via
        # _resolve where read (node_form / node_catalogue).
        param_spec=param_spec if callable(param_spec) else dict(param_spec or {}),
    )
    _NODE_REGISTRY[stage] = node
    return node


def registered_stage_names() -> frozenset:
    """Every registered node-type name. Single source of truth for the DAG vocabulary —
    used by introspection, the graph preview, and the architecture-doc drift guard
    (``tests/test_node_catalogue_doc.py``)."""
    return frozenset(_NODE_REGISTRY)


def stages_in_category(category: str) -> frozenset:
    """The registered stage names whose declared ``category`` matches (taxonomy.py). The
    single way to ask "which nodes are models / metrics / sinks …" — replaces hardcoded
    stage lists. ``stages_in_category("model")`` is the device/offload-managed set.

    Note: an operator with a *callable* category (e.g. ``convert``: model for asr, transform
    for tts) is excluded here — resolve a concrete node's category via :func:`node_category`."""
    return frozenset(s for s, d in _NODE_REGISTRY.items() if d.category == category)


def node_category(stage: str, params: Optional[dict] = None) -> str:
    """A node instance's declared category, resolving a callable (operator) category."""
    return _resolve(get_stage_node_def(stage).category, params)


def node_domain(stage: str, params: Optional[dict] = None) -> str:
    """A node instance's declared domain, resolving a callable (operator) domain."""
    return _resolve(get_stage_node_def(stage).domain, params)


def node_model_field(stage: str, params: Optional[dict] = None) -> Optional[str]:
    """The config key selecting a node instance's model, resolving a callable model_field
    (``embed`` → text/audio emb type by modality; ``convert`` → asr type or None for tts)."""
    return _resolve(get_stage_node_def(stage).model_field, params)


def model_node_kinds() -> frozenset:
    """Legacy node-kind names of every model-category node type — the set the device map
    (``executor/parallel.py``) must cover. Resolves an operator to its model sub-kinds
    (``embed`` → text/audio/corpus_embedding, keeping the model-category ones), so the
    device self-check still auto-detects a new model node missing a placement after the
    collapse (when ``stages_in_category('model')`` would just return ``{'embed', …}``)."""
    from .operators import ALIASES, is_operator

    kinds: set = set()
    for t in _NODE_REGISTRY:
        if is_operator(t):
            for old, (op, fixed) in ALIASES.items():
                if op == t and node_category(t, fixed) == "model":
                    kinds.add(old)
        elif node_category(t) == "model":
            kinds.add(t)
    return frozenset(kinds)


def get_stage_node_def(stage: str) -> StageNodeDef:
    if stage not in _NODE_REGISTRY:
        registered = ", ".join(sorted(_NODE_REGISTRY))
        hint = (
            " (corpus_index was replaced by corpus_embedding + vector_db)"
            if stage == "corpus_index"
            else ""
        )
        raise KeyError(f"Unknown stage node: {stage}{hint}. Registered: {registered}")
    return _NODE_REGISTRY[stage]


# Node catalogue. A node that runs a model declares the config field that selects it
# (required-fields validation + per-stage lifecycle derive from one place); inputs/outputs
# declare its data contract (edge validation). metrics has no required input — it appears
# in asr_only too and adapts to whatever ran.
#
# dataset_source is the graph root: it surfaces + validates the dataset's source
# artifacts so every downstream node hangs off it. (Actual loading + TTS synthesis stay
# in prepare_dataset before the graph — needed for mode detection and to keep TTS off the
# embedder's device; this node represents that source in the executed graph.)
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


register_stage_node(
    "combine",
    category="transform",
    domain=_combine_domain,
    inputs=_combine_inputs,
    outputs=_combine_outputs,
    optional_inputs=_combine_optional,
    param_spec={
        "level": {"kind": "select", "choices": ["embedding", "result", "set"],
                  "default": "embedding"},
        # result-fusion params (the rank-fusion of two retrieved sets)
        "hybrid": {"kind": "bool", "default": False, "show_if": {"level": ["result"]}},
        "method": {"kind": "select", "choices": ["rrf", "weighted", "max_score"],
                   "default": "rrf", "show_if": {"level": ["result"]}},
        "weight": {"kind": "number", "default": 0.5, "show_if": {"level": ["result"]}},
        "k": {"kind": "number", "show_if": {"level": ["result"]}},
        "top_k": {"kind": "number", "show_if": {"level": ["result"]}},
        "rrf_k": {"kind": "number", "default": 60, "show_if": {"level": ["result"]}},
    },
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


register_stage_node(
    "search",
    category="transform",
    domain="retrieval",
    inputs=_search_inputs,
    outputs=(ARTIFACT_RETRIEVED,),
    optional_inputs=_search_optional,
    param_spec={
        "k": {"kind": "number", "default": 5},
        "mode": {"kind": "select", "choices": ["dense", "sparse", "hybrid"],
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
            "kind": "select", "choices": ["rrf", "weighted", "union", "intersection"],
            "default": "rrf", "show_if": {"method": ["multi_query", "decompose"]}},
    },
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
                 "choices": ["none", "token_overlap", "cross_encoder"],
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
    "judge": {"inputs": (ARTIFACT_METRICS,), "outputs": (ARTIFACT_JUDGE_SCORES,),
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


def validate_graph_artifacts(graph: "StageGraph") -> None:
    """Check every node's required inputs are satisfiable in topological order.

    An input is satisfied if it is a SOURCE artifact or produced by an upstream node.
    Raises ValueError naming the first unsatisfiable input (a wiring bug).
    """
    available = set(SOURCE_ARTIFACTS)
    for level in graph.topological_levels():
        produced_this_level: List[str] = []
        for node in level:
            node_def = get_stage_node_def(node.stage)
            missing = []
            for a in _resolve(node_def.inputs, node.params):
                if isinstance(a, OneOf):
                    if not any(alt in available for alt in a):
                        missing.append(a.key or tuple(a))
                elif a not in available:
                    missing.append(a)
            if missing:
                raise ValueError(
                    f"Stage '{node.id}' needs artifact(s) {missing} that no upstream "
                    f"node or source provides (mode={graph.mode})."
                )
            produced_this_level.extend(_effective_outputs(node.stage, node.params))
        available.update(produced_this_level)
