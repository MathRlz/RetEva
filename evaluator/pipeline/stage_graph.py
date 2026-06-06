"""DAG-lite stage graph for current pipeline modes.

Two registries drive graph construction:

* ``register_stage_node`` — declares a node *type* (its name + the config field of
  the model it runs, if any). This is the structural twin of the executable handler
  registry in ``evaluation/stage_registry.py``: a node declares itself once and
  introspection (model lifecycle, required-field derivation) reads it, instead of a
  hardcoded per-mode map.
* ``PIPELINE_MODE_SPECS`` — named modes are now thin *presets*: an ordered sequence
  of node names (a tuple element = a parallel level) chained into a DAG by
  ``_chain``. Adding a node to a mode is a one-line edit to its sequence.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union


@dataclass(frozen=True)
class StageNode:
    """Single stage node in the execution DAG."""

    id: str
    stage: str
    depends_on: Tuple[str, ...] = ()


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


@dataclass(frozen=True)
class StageNodeDef:
    """Declared node *type*: its name + the config field of the model it runs."""

    stage: str
    model_field: Optional[str] = None  # e.g. "model.asr_model_type"; None = model-free


_NODE_REGISTRY: Dict[str, StageNodeDef] = {}


def register_stage_node(
    stage: str, *, model_field: Optional[str] = None
) -> StageNodeDef:
    """Register a node type. ``model_field`` names the config key for its model."""
    if stage in _NODE_REGISTRY:
        raise ValueError(f"Stage node already registered: {stage}")
    node = StageNodeDef(stage=stage, model_field=model_field)
    _NODE_REGISTRY[stage] = node
    return node


def get_stage_node_def(stage: str) -> StageNodeDef:
    if stage not in _NODE_REGISTRY:
        registered = ", ".join(sorted(_NODE_REGISTRY))
        raise KeyError(f"Unknown stage node: {stage}. Registered: {registered}")
    return _NODE_REGISTRY[stage]


# Node catalogue. A node that runs a model declares the config field that selects it,
# so required-fields validation and per-stage model lifecycle derive from one place.
register_stage_node("asr", model_field="model.asr_model_type")
register_stage_node("text_embedding", model_field="model.text_emb_model_type")
register_stage_node("audio_embedding", model_field="model.audio_emb_model_type")
register_stage_node("fusion")
register_stage_node("retrieval")
register_stage_node("metrics")
register_stage_node("answer_gen")
register_stage_node("finalize")


# A preset element is a stage name, or a tuple of names that form one parallel level.
PresetElement = Union[str, Tuple[str, ...]]
NodeSequence = Sequence[PresetElement]


def _chain(sequence: NodeSequence) -> Tuple[StageNode, ...]:
    """Build a DAG from an ordered sequence; each element depends on the previous one.

    A tuple element becomes a parallel level (all its nodes share the prior element as
    parents, and the next element depends on all of them) — that is how fusion's two
    embedding sources are expressed.
    """
    nodes: List[StageNode] = []
    prev_ids: Tuple[str, ...] = ()
    for element in sequence:
        group = (element,) if isinstance(element, str) else tuple(element)
        for stage in group:
            get_stage_node_def(stage)  # validate the node is registered
            nodes.append(StageNode(id=stage, stage=stage, depends_on=prev_ids))
        prev_ids = group
    return tuple(nodes)


def _audio_text_sequence(embedding_fusion_enabled: bool) -> NodeSequence:
    if embedding_fusion_enabled:
        return [
            ("audio_embedding", "text_embedding"),
            "fusion",
            "retrieval",
            "metrics",
            "answer_gen",
            "finalize",
        ]
    return ["audio_embedding", "retrieval", "metrics", "answer_gen", "finalize"]


@dataclass(frozen=True)
class PipelineModeSpec:
    mode: str
    required_model_fields: Tuple[str, ...]
    build_nodes: Callable[[bool], Tuple[StageNode, ...]]


# Named modes = thin presets selecting an ordered node sequence (chained into a DAG).
_MODE_SEQUENCES: Dict[str, Callable[[bool], NodeSequence]] = {
    "asr_only": lambda _fusion: ["asr", "metrics", "finalize"],
    "asr_text_retrieval": lambda _fusion: [
        "asr",
        "text_embedding",
        "retrieval",
        "metrics",
        "answer_gen",
        "finalize",
    ],
    "audio_emb_retrieval": lambda _fusion: [
        "audio_embedding",
        "retrieval",
        "metrics",
        "answer_gen",
        "finalize",
    ],
    "audio_text_retrieval": _audio_text_sequence,
}


def _required_model_fields(mode: str) -> Tuple[str, ...]:
    """Union of model fields over a mode's nodes (fusion on, the maximal node set)."""
    sequence = _MODE_SEQUENCES[mode](True)
    fields: List[str] = []
    for element in sequence:
        group = (element,) if isinstance(element, str) else tuple(element)
        for stage in group:
            field = get_stage_node_def(stage).model_field
            if field and field not in fields:
                fields.append(field)
    return tuple(fields)


def _make_spec(mode: str) -> PipelineModeSpec:
    sequence_for = _MODE_SEQUENCES[mode]
    return PipelineModeSpec(
        mode=mode,
        required_model_fields=_required_model_fields(mode),
        build_nodes=lambda fusion_enabled: _chain(sequence_for(fusion_enabled)),
    )


PIPELINE_MODE_SPECS: Dict[str, PipelineModeSpec] = {
    mode: _make_spec(mode) for mode in _MODE_SEQUENCES
}


def list_pipeline_mode_specs() -> List[PipelineModeSpec]:
    return [PIPELINE_MODE_SPECS[key] for key in sorted(PIPELINE_MODE_SPECS.keys())]


def resolve_pipeline_mode_spec(mode: str) -> PipelineModeSpec:
    if mode not in PIPELINE_MODE_SPECS:
        available = ", ".join(sorted(PIPELINE_MODE_SPECS.keys()))
        raise ValueError(f"Unknown pipeline mode: {mode}. Available modes: {available}")
    return PIPELINE_MODE_SPECS[mode]


def build_stage_graph(
    mode: str, *, embedding_fusion_enabled: bool = False
) -> StageGraph:
    """Build execution DAG for currently supported pipeline modes."""
    spec = resolve_pipeline_mode_spec(mode)
    nodes = spec.build_nodes(embedding_fusion_enabled)
    return StageGraph(mode=mode, nodes=nodes)
