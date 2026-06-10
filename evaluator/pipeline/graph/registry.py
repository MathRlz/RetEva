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


# ── Artifact vocabulary (Data-as-DAG phase 1) ─────────────────────────
# Named, typed values passed between nodes. Edges connect a producer's output to a
# consumer's input *by artifact name*. SOURCE_ARTIFACTS are provided externally by the
# dataset / runtime (so a node can require them without an upstream producer).
#
# Note: ``query_text`` is both a source (the dataset's question text) *and* produced by
# ``asr`` (the hypotheses overwrite it). That duality is why text embedding works in
# ``audio_text_retrieval`` (text comes from the dataset) and in ``asr_text_retrieval``
# (text comes from ASR) with one declared input.
ARTIFACT_QUERY_AUDIO = "query_audio"
ARTIFACT_QUERY_TEXT = "query_text"
# Raw (pre-correction/optimization) ASR output — distinct from query_text, which becomes the
# *effective* query after a correction/augmentation branch. WER/CER score against this (T4/L1).
ARTIFACT_RAW_QUERY_TEXT = "raw_query_text"
ARTIFACT_CORPUS = "corpus"
ARTIFACT_RELEVANT_DOCS = "relevant_docs"
ARTIFACT_SHORT_ANSWERS = "short_answers"
ARTIFACT_QUERY_VECTORS = "query_vectors"
ARTIFACT_VECTOR_INDEX = "vector_index"
ARTIFACT_RETRIEVED = "retrieved"
ARTIFACT_METRICS = "metrics"
ARTIFACT_GENERATED_ANSWERS = "generated_answers"

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

_DATASET_SOURCE_ROLE_OUTPUTS: Dict[str, Tuple[str, ...]] = {
    DATASET_ROLE_CORPUS: (ARTIFACT_CORPUS,),
    DATASET_ROLE_QUESTIONS: (
        ARTIFACT_QUERY_AUDIO,
        ARTIFACT_QUERY_TEXT,
        ARTIFACT_RELEVANT_DOCS,
        ARTIFACT_SHORT_ANSWERS,
    ),
}


def _effective_outputs(stage: str, params: Optional[dict]) -> Tuple[str, ...]:
    """Outputs a node instance advertises, honoring a ``dataset_source`` ``role`` param.

    Non-``dataset_source`` nodes always advertise their registered outputs. A
    ``dataset_source`` with ``role: corpus``/``questions`` advertises only that slice;
    ``both`` (default / unset) advertises the full set."""
    ndef = get_stage_node_def(stage)
    if stage != "dataset_source":
        return ndef.outputs
    role = (params or {}).get("role") or DATASET_ROLE_BOTH
    if role == DATASET_ROLE_BOTH:
        return ndef.outputs
    if role not in _DATASET_SOURCE_ROLE_OUTPUTS:
        allowed = sorted([*_DATASET_SOURCE_ROLE_OUTPUTS, DATASET_ROLE_BOTH])
        raise ValueError(f"Unknown dataset_source role '{role}'. Allowed: {allowed}")
    return _DATASET_SOURCE_ROLE_OUTPUTS[role]


@dataclass(frozen=True)
class StageNodeDef:
    """Declared node *type*: name, the config field of the model it runs, and the
    artifacts it consumes (``inputs``) / produces (``outputs``)."""

    stage: str
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


_NODE_REGISTRY: Dict[str, StageNodeDef] = {}


def register_stage_node(
    stage: str,
    *,
    model_field: Optional[str] = None,
    inputs: Tuple[str, ...] = (),
    outputs: Tuple[str, ...] = (),
    optional_inputs: Tuple[str, ...] = (),
    param_defaults: Optional[Mapping[str, Any]] = None,
) -> StageNodeDef:
    """Register a node type. ``model_field`` names the config key for its model;
    ``inputs``/``outputs`` are the required-consumed/produced artifacts; ``optional_inputs``
    are wired to their producer only when present (ordering / optional data). ``param_defaults``
    lets CSE treat an explicit-default param as an omitted one (S7)."""
    if stage in _NODE_REGISTRY:
        raise ValueError(f"Stage node already registered: {stage}")
    node = StageNodeDef(
        stage=stage,
        model_field=model_field,
        inputs=inputs,
        outputs=outputs,
        optional_inputs=optional_inputs,
        param_defaults=dict(param_defaults or {}),
    )
    _NODE_REGISTRY[stage] = node
    return node


def get_stage_node_def(stage: str) -> StageNodeDef:
    if stage not in _NODE_REGISTRY:
        registered = ", ".join(sorted(_NODE_REGISTRY))
        raise KeyError(f"Unknown stage node: {stage}. Registered: {registered}")
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
register_stage_node(
    "dataset_source",
    inputs=(),
    outputs=(
        ARTIFACT_QUERY_AUDIO,
        ARTIFACT_QUERY_TEXT,
        ARTIFACT_CORPUS,
        ARTIFACT_RELEVANT_DOCS,
        ARTIFACT_SHORT_ANSWERS,
    ),
)
# TTS: synthesize query audio from the dataset's text (TTS-bridge for text datasets).
# Consumes query_text, produces query_audio so asr/audio_embedding wire after it.
register_stage_node(
    "tts",
    inputs=(ARTIFACT_QUERY_TEXT,),
    outputs=(ARTIFACT_QUERY_AUDIO,),
)
register_stage_node(
    "asr",
    model_field="model.asr_model_type",
    inputs=(ARTIFACT_QUERY_AUDIO,),
    outputs=(ARTIFACT_QUERY_TEXT, ARTIFACT_RAW_QUERY_TEXT),
)
register_stage_node(
    "text_embedding",
    model_field="model.text_emb_model_type",
    inputs=(ARTIFACT_QUERY_TEXT,),
    outputs=(ARTIFACT_QUERY_VECTORS,),
)
register_stage_node(
    "audio_embedding",
    model_field="model.audio_emb_model_type",
    inputs=(ARTIFACT_QUERY_AUDIO,),
    outputs=(ARTIFACT_QUERY_VECTORS,),
)
register_stage_node(
    "fusion",
    inputs=(ARTIFACT_QUERY_VECTORS,),
    outputs=(ARTIFACT_QUERY_VECTORS,),
)
# LLM query optimization (rewrite / HyDE / multi-query): transforms the query text
# before embedding. Consumes + produces query_text so text_embedding wires after it.
register_stage_node(
    "query_optimization",
    inputs=(ARTIFACT_QUERY_TEXT,),
    outputs=(ARTIFACT_QUERY_TEXT,),
)
# Post-ASR domain correction (drug/dose/unit repair): transforms the query text before
# embedding. Consumes + produces query_text so it chains after asr, before query_optimization.
register_stage_node(
    "query_correction",
    inputs=(ARTIFACT_QUERY_TEXT,),
    outputs=(ARTIFACT_QUERY_TEXT,),
)
# Robustness perturbation (C2): corrupts the query text (ASR-confusion homophones / dangerous
# dose-unit swaps) — a branch-divergence source for robustness experiments. Consumes + produces
# query_text so it chains in the asr→…→embed path like correction. Configured per node/branch via
# params (deterministic, seeded per item).
register_stage_node(
    "augmenter",
    inputs=(ARTIFACT_QUERY_TEXT,),
    outputs=(ARTIFACT_QUERY_TEXT,),
)
# Corpus side of the graph: embed the corpus (text or audio) and build the vector
# index. A root node (corpus is a source artifact) that feeds retrieval. Its model is
# the same embedder instance the query side uses (offload planner accounts for both).
register_stage_node(
    "corpus_index",
    inputs=(ARTIFACT_CORPUS,),
    outputs=(ARTIFACT_VECTOR_INDEX,),
)
register_stage_node(
    "retrieval",
    inputs=(ARTIFACT_QUERY_VECTORS, ARTIFACT_VECTOR_INDEX),
    outputs=(ARTIFACT_RETRIEVED,),
)
# Post-retrieval refinement (cross-encoder/token rerank + MMR + threshold). Present
# only when refinement is configured; consumes + produces retrieved so metrics/
# answer_gen wire to it (the refined results) when it runs.
register_stage_node(
    "rerank",
    inputs=(ARTIFACT_RETRIEVED,),
    outputs=(ARTIFACT_RETRIEVED,),
)
# metrics/answer_gen/finalize use optional_inputs to encode ordering: metrics after
# retrieval (when retrieving) and after asr (asr_only); answer_gen after metrics (it reads
# metrics intermediates); finalize last (after answer_gen when present).
register_stage_node(
    "metrics",
    inputs=(),
    outputs=(ARTIFACT_METRICS,),
    optional_inputs=(ARTIFACT_RETRIEVED, ARTIFACT_QUERY_TEXT, ARTIFACT_RAW_QUERY_TEXT),
)
register_stage_node(
    "answer_gen",
    inputs=(ARTIFACT_RETRIEVED,),
    outputs=(ARTIFACT_GENERATED_ANSWERS,),
    optional_inputs=(ARTIFACT_METRICS,),
)
register_stage_node(
    "finalize",
    inputs=(ARTIFACT_METRICS,),
    outputs=(),
    optional_inputs=(ARTIFACT_GENERATED_ANSWERS,),
)
# Persist the run's generated/synthesized data (prepared audio dataset, generated
# answers) — a terminal leaf that runs after its producers (tts / answer_gen).
register_stage_node(
    "dataset_sink",
    inputs=(),
    outputs=(),
    optional_inputs=(ARTIFACT_QUERY_AUDIO, ARTIFACT_GENERATED_ANSWERS),
)
# Aggregate: terminal report builder. Scans every branch's `retrieved` (per-producer keyed)
# + the shared ground-truth artifacts → per-branch metrics + cross-branch deltas (W6/A9).
# optional_inputs order it after retrieval + metrics.
register_stage_node(
    "aggregate",
    inputs=(),
    outputs=(),
    optional_inputs=(ARTIFACT_RETRIEVED, ARTIFACT_METRICS),
)
# Terminal sinks: persist the aggregate report (A6). They run after metrics (which produces
# the report); ordering via optional_inputs on metrics output.
register_stage_node(
    "leaderboard_sink",
    inputs=(),
    outputs=(),
    optional_inputs=(ARTIFACT_METRICS,),
)
register_stage_node(
    "tracking_sink",
    inputs=(),
    outputs=(),
    optional_inputs=(ARTIFACT_METRICS,),
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
            missing = [a for a in node_def.inputs if a not in available]
            if missing:
                raise ValueError(
                    f"Stage '{node.id}' needs artifact(s) {missing} that no upstream "
                    f"node or source provides (mode={graph.mode})."
                )
            produced_this_level.extend(_effective_outputs(node.stage, node.params))
        available.update(produced_this_level)
