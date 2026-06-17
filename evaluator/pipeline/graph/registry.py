"""Node-type + stage-graph registry: the DAG's structural vocabulary.

``register_stage_node`` declares a node *type* (its name + the config field of the model
it runs, if any). This is the structural twin of the executable handler registry in
``evaluation/stage_registry.py``: a node declares itself once and introspection (model
lifecycle, required-field derivation) reads it, instead of a hardcoded per-mode map.

Also hosts the ``StageNode`` / ``StageGraph`` data structures and
``validate_graph_artifacts`` (satisfiability check over topological order). The artifact
vocabulary lives in ``artifacts.py`` (re-exported below so it stays importable from here);
the operator catalogue — the 11 ``register_stage_node`` calls — lives in
``operators_catalog.py``, imported at the bottom so importing this module populates the
node registry.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

from .taxonomy import validate_taxonomy

# Re-export the artifact vocabulary so every name historically importable from this module
# (the package __init__ and several importers do `from .registry import ...`) keeps
# resolving here after the split.
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
    _DOCS_AXIS_CAPABLE,
    display_artifact_names,
    one_of,
)


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


# Importing the operator catalogue runs the 11 ``register_stage_node(...)`` calls (in order),
# populating ``_NODE_REGISTRY``. It must come AFTER ``register_stage_node`` + the machinery
# above are defined, since ``operators_catalog`` imports from this module at its top.
from . import operators_catalog  # noqa: E402,F401  (triggers node registration)

# Third-party node-type plugins (entry-point group ``evaluator.nodes``, §5) — best-effort.
try:
    from ...plugins import load_plugins as _load_node_plugins

    _load_node_plugins("evaluator.nodes")
except Exception:  # pragma: no cover - discovery never breaks core import
    pass
