"""Node-type + stage-graph registry: the DAG's structural vocabulary.

Hosts ``StageNode``/``StageGraph`` and ``validate_graph_artifacts``; ``operators_catalog.py``
is imported at the bottom so importing this module populates the registry.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

from .taxonomy import validate_taxonomy
from .operators import ALIASES, expand_alias, is_operator, node_kind
from ..artifacts import artifact_dtype, artifact_modality, is_registered

# Re-exported so every name historically importable from this module keeps resolving here.
from .artifacts import (  # noqa: F401  (re-exported vocabulary)
    ARTIFACT_ANSWER_SCORES,
    ARTIFACT_AUDIO_QUERY_VECTORS,
    ARTIFACT_AUGMENTED_QUERY_TEXT,
    ARTIFACT_CORPUS,
    ARTIFACT_CORPUS_VECTORS,
    ARTIFACT_CORRECTED_QUERY_TEXT,
    ARTIFACT_EMBEDDING_ALIGNMENT,
    ARTIFACT_FUSED_QUERY_VECTORS,
    ARTIFACT_GENERATED_ANSWERS,
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
    DATASET_ROLE_BOTH,
    DATASET_ROLE_CORPUS,
    DATASET_ROLE_QUESTIONS,
    QUERY_TEXT_CHAIN,
    SOURCE_ARTIFACTS,
    OneOf,
    _DATASET_SOURCE_ROLE_OUTPUTS,
    DOCS_AXIS_CAPABLE,
    display_artifact_names,
    one_of,
)


@dataclass(frozen=True)
class StageNode:
    """Single stage node in the execution DAG. ``bindings`` maps each consumed artifact to its
    producer node id(s) in producer order (several when e.g. fusion's two embedders both feed
    it); ``depends_on`` is the set of producer ids and is the contract for topological sort."""

    id: str
    stage: str
    depends_on: Tuple[str, ...] = ()
    bindings: Tuple[Tuple[str, str], ...] = ()
    # Canonical-key -> ordered candidate artifact names for OneOf inputs; empty for plain
    # inputs. ``s.input(key)`` reads the highest-priority candidate a bound producer published.
    input_aliases: Tuple[Tuple[str, Tuple[str, ...]], ...] = ()
    # Per-instance config (duplicate/arbitrary nodes); None for single-instance modes.
    params: Optional[dict] = field(default=None, compare=False, hash=False)


@dataclass(frozen=True)
class StageGraph:
    """Execution graph with deterministic topological levels."""

    mode: str
    nodes: Tuple[StageNode, ...]

    def node_ids(self) -> List[str]:
        return [node.id for node in self.nodes]

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


def _resolve(field: Any, params: Optional[dict]) -> Any:
    """Resolve a value-or-callable ``StageNodeDef`` field — lets one generic operator vary its
    contract by field (e.g. ``embed`` producing ``corpus_vectors`` vs ``text_query_vectors``
    by ``axis``); a no-op identity for static registrations."""
    return field(params or {}) if callable(field) else field


def effective_outputs(stage: str, params: Optional[dict]) -> Tuple[str, ...]:
    """Outputs a node instance advertises, honoring dataset_source ``role``/``fields``: non-
    dataset_source nodes advertise their registered outputs; dataset_source with a ``fields``
    param advertises exactly those, narrowed by ``role`` (``both``, the default, keeps the set)."""
    ndef = get_stage_node_def(stage)
    # `axis: docs` flips a type-preserving T->T transform to the corpus instead of query_text.
    if stage in DOCS_AXIS_CAPABLE and (params or {}).get("axis") == "docs":
        return (ARTIFACT_CORPUS,)

    # Applies whether authored as the legacy name or the `source` operator (node_kind reverses
    # union:false -> dataset_source).
    if node_kind(stage, params) != "dataset_source":
        return _resolve(ndef.outputs, params)
    fields = (params or {}).get("fields") or {}
    # Derived/bridge artifacts the column schema can't name (self-retrieval corpus, TTS-bridge
    # reference_transcription), advertised so the preview wires what the run wires statically.
    extra = (params or {}).get("extra_outputs") or ()
    # A same-graph transform that re-produces an artifact (e.g. tts synthesizing query_audio) is
    # its real producer; suppress it here or the port dangles and the source shows an input the
    # dataset lacks. Injected by ``_attach_dataset_fields`` (it has the whole node list).
    suppress = set((params or {}).get("suppress_outputs") or ())
    _full: Tuple[str, ...] = _resolve(ndef.outputs, params)
    base: Tuple[str, ...] = _full
    if fields or extra:
        # Dedupe, keeping registered-output order first (stable wiring), then any extras.
        declared = list(dict.fromkeys(list(fields.values()) + list(extra)))
        base = tuple(
            [a for a in _full if a in declared]
            + [a for a in declared if a not in _full]
        )
    if suppress:
        base = tuple(a for a in base if a not in suppress)
    role = (params or {}).get("role") or DATASET_ROLE_BOTH
    if role == DATASET_ROLE_BOTH:
        return base
    if role not in _DATASET_SOURCE_ROLE_OUTPUTS:
        allowed = sorted([*_DATASET_SOURCE_ROLE_OUTPUTS, DATASET_ROLE_BOTH])
        raise ValueError(f"Unknown dataset_source role '{role}'. Allowed: {allowed}")
    narrowed = tuple(a for a in _DATASET_SOURCE_ROLE_OUTPUTS[role] if a not in suppress)
    if fields:
        # An empty role/fields intersection means the schema doesn't describe this usage;
        # the role stays authoritative.
        inter = tuple(a for a in base if a in narrowed)
        return inter if inter else narrowed
    return narrowed


def effective_inputs(stage: str, params: Optional[dict]) -> Tuple[str, ...]:
    """Required inputs a node instance consumes (display + wiring truthfulness); a docs-axis
    transform consumes the corpus, not query_text."""
    ndef = get_stage_node_def(stage)
    if stage in DOCS_AXIS_CAPABLE and (params or {}).get("axis") == "docs":
        return (ARTIFACT_CORPUS,)
    return display_artifact_names(_resolve(ndef.inputs, params))


def dataset_columns(params: Optional[dict]) -> List[Dict[str, str]]:
    """The display schema of a dataset_source node instance: ``[{name, artifact, type, dtype}]``,
    built from the injected ``params.fields`` + the artifact registry. Empty when the node
    carries no schema."""

    fields = (params or {}).get("fields") or {}
    columns: List[Dict[str, str]] = []
    for name, artifact in fields.items():
        registered = is_registered(str(artifact))
        columns.append(
            {
                "name": str(name),
                "artifact": str(artifact),
                "type": str(artifact_modality(str(artifact)).value) if registered else "?",
                "dtype": artifact_dtype(str(artifact)) if registered else "?",
            }
        )
    return columns


@dataclass(frozen=True)
class StageNodeDef:
    """Declared node *type*: name, the config field of the model it runs, and the
    artifacts it consumes (``inputs``) / produces (``outputs``)."""

    stage: str
    # `category` is the data-flow role, load-bearing for lifecycle/validation/execution;
    # `domain` is the functional area (UI grouping/DAG coloring/docs). Both required.
    category: str = "transform"
    domain: str = "ingest"
    model_field: Optional[str] = None  # e.g. "model.asr_model_type"; None = model-free
    inputs: Tuple[str, ...] = ()  # validated; gates execution
    outputs: Tuple[str, ...] = ()
    # Ordering/data that may be absent (e.g. metrics runs after retrieval only when present);
    # auto-wiring adds a producer dependency only when the graph actually produces it.
    optional_inputs: Tuple[str, ...] = ()
    # A param explicitly set to its default is canonically identical to omitting it, so CSE
    # collapses an explicit-vs-default twin into one run.
    param_defaults: Mapping[str, Any] = field(default_factory=dict)
    # Node-scoped builder params, NOT model params (those come from the model's registry
    # `Params` schema) — only genuinely node-level switches belong here (e.g. asr.oracle).
    param_spec: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)


NODE_REGISTRY: Dict[str, StageNodeDef] = {}


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
    """Register a node type. ``category``/``domain`` (taxonomy.py role + area) are required and
    validated. ``param_defaults`` lets CSE treat an explicit-default param as an omitted one."""
    if stage in NODE_REGISTRY:
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
        # May be a callable(params) for a field-aware builder form (e.g. transform's `method`
        # choices differ by `op`); resolved via _resolve where read.
        param_spec=param_spec if callable(param_spec) else dict(param_spec or {}),
    )
    NODE_REGISTRY[stage] = node
    return node


def registered_stage_names() -> frozenset:
    """Every registered node-type name. Single source of truth for the DAG vocabulary —
    used by introspection, the graph preview, and the architecture-doc drift guard
    (``tests/test_node_catalogue_doc.py``)."""
    return frozenset(NODE_REGISTRY)


def stages_in_category(category: str) -> frozenset:
    """Registered stage names whose declared ``category`` matches (taxonomy.py); the single way
    to ask "which nodes are models/metrics/sinks", replacing hardcoded stage lists.

    Note: an operator with a *callable* category (e.g. ``convert``) is excluded — resolve a
    concrete node's category via :func:`node_category`."""
    return frozenset(s for s, d in NODE_REGISTRY.items() if d.category == category)


def node_category(stage: str, params: Optional[dict] = None) -> str:
    """A node instance's declared category, resolving a callable (operator) category."""
    return _resolve(get_stage_node_def(stage).category, params)


def node_domain(stage: str, params: Optional[dict] = None) -> str:
    """A node instance's declared domain, resolving a callable (operator) domain."""
    return _resolve(get_stage_node_def(stage).domain, params)


def is_structural(stage: str, params: Optional[dict] = None) -> bool:
    """Whether a node is plumbing the user never authors (``measure``/``sink``), auto-derived
    and hidden from the builder palette — except the LLM judge (``answer_judge``), a
    user-facing Model, not plumbing."""

    operator, _eff = expand_alias(stage, dict(params or {}))
    if operator not in ("measure", "sink"):
        return False
    return node_kind(stage, params) != "answer_judge"


def node_model_field(stage: str, params: Optional[dict] = None) -> Optional[str]:
    """The config key selecting a node instance's model, resolving a callable model_field
    (``embed`` → text/audio emb type by modality; ``convert`` → asr type or None for tts)."""
    return _resolve(get_stage_node_def(stage).model_field, params)


def model_node_kinds() -> frozenset:
    """Legacy node-kind names of every model-category node type — the set the device map
    (``executor/parallel.py``) must cover, resolved so the device self-check still auto-detects
    a new model node missing a placement after the operator collapse."""

    kinds: set = set()
    for t in NODE_REGISTRY:
        if is_operator(t):
            for old, (op, fixed) in ALIASES.items():
                if op == t and node_category(t, fixed) == "model":
                    kinds.add(old)
        elif node_category(t) == "model":
            kinds.add(t)
    return frozenset(kinds)


def get_stage_node_def(stage: str) -> StageNodeDef:
    if stage not in NODE_REGISTRY:
        registered = ", ".join(sorted(NODE_REGISTRY))
        hint = (
            " (corpus_index was replaced by corpus_embedding + vector_db)"
            if stage == "corpus_index"
            else ""
        )
        raise KeyError(f"Unknown stage node: {stage}{hint}. Registered: {registered}")
    return NODE_REGISTRY[stage]


def validate_graph_artifacts(graph: "StageGraph") -> None:
    """Check every node's required inputs are satisfiable in topological order: an input is
    satisfied by a SOURCE artifact or an upstream producer. Raises, naming the first
    unsatisfiable input."""
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
            produced_this_level.extend(effective_outputs(node.stage, node.params))
        available.update(produced_this_level)


# Must come after `register_stage_node` + the machinery above, since `operators_catalog`
# imports from this module at its top.
from . import operators_catalog  # noqa: E402,F401  (triggers node registration)
