"""DAG-lite stage graph for current pipeline modes."""

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple


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
class PipelineModeSpec:
    mode: str
    required_model_fields: Tuple[str, ...]
    build_nodes: Callable[[bool], Tuple[StageNode, ...]]


def _nodes_asr_only(_embedding_fusion_enabled: bool) -> Tuple[StageNode, ...]:
    return (
        StageNode(id="asr", stage="asr"),
        StageNode(id="metrics", stage="metrics", depends_on=("asr",)),
    )


def _nodes_asr_text_retrieval(_embedding_fusion_enabled: bool) -> Tuple[StageNode, ...]:
    return (
        StageNode(id="asr", stage="asr"),
        StageNode(id="text_embedding", stage="text_embedding", depends_on=("asr",)),
        StageNode(id="retrieval", stage="retrieval", depends_on=("text_embedding",)),
        StageNode(id="metrics", stage="metrics", depends_on=("retrieval",)),
    )


def _nodes_audio_emb_retrieval(_embedding_fusion_enabled: bool) -> Tuple[StageNode, ...]:
    return (
        StageNode(id="audio_embedding", stage="audio_embedding"),
        StageNode(id="retrieval", stage="retrieval", depends_on=("audio_embedding",)),
        StageNode(id="metrics", stage="metrics", depends_on=("retrieval",)),
    )


def _nodes_audio_text_retrieval(embedding_fusion_enabled: bool) -> Tuple[StageNode, ...]:
    if embedding_fusion_enabled:
        return (
            StageNode(id="audio_embedding", stage="audio_embedding"),
            StageNode(id="text_embedding", stage="text_embedding"),
            StageNode(
                id="fusion",
                stage="fusion",
                depends_on=("audio_embedding", "text_embedding"),
            ),
            StageNode(id="retrieval", stage="retrieval", depends_on=("fusion",)),
            StageNode(id="metrics", stage="metrics", depends_on=("retrieval",)),
        )
    return (
        StageNode(id="audio_embedding", stage="audio_embedding"),
        StageNode(id="retrieval", stage="retrieval", depends_on=("audio_embedding",)),
        StageNode(id="metrics", stage="metrics", depends_on=("retrieval",)),
    )


PIPELINE_MODE_SPECS: Dict[str, PipelineModeSpec] = {
    "asr_only": PipelineModeSpec(
        mode="asr_only",
        required_model_fields=("model.asr_model_type",),
        build_nodes=_nodes_asr_only,
    ),
    "asr_text_retrieval": PipelineModeSpec(
        mode="asr_text_retrieval",
        required_model_fields=("model.asr_model_type", "model.text_emb_model_type"),
        build_nodes=_nodes_asr_text_retrieval,
    ),
    "audio_emb_retrieval": PipelineModeSpec(
        mode="audio_emb_retrieval",
        required_model_fields=("model.audio_emb_model_type",),
        build_nodes=_nodes_audio_emb_retrieval,
    ),
    "audio_text_retrieval": PipelineModeSpec(
        mode="audio_text_retrieval",
        required_model_fields=("model.audio_emb_model_type", "model.text_emb_model_type"),
        build_nodes=_nodes_audio_text_retrieval,
    ),
}


def list_pipeline_mode_specs() -> List[PipelineModeSpec]:
    return [PIPELINE_MODE_SPECS[key] for key in sorted(PIPELINE_MODE_SPECS.keys())]


def resolve_pipeline_mode_spec(mode: str) -> PipelineModeSpec:
    if mode not in PIPELINE_MODE_SPECS:
        available = ", ".join(sorted(PIPELINE_MODE_SPECS.keys()))
        raise ValueError(f"Unknown pipeline mode: {mode}. Available modes: {available}")
    return PIPELINE_MODE_SPECS[mode]


def build_stage_graph(mode: str, *, embedding_fusion_enabled: bool = False) -> StageGraph:
    """Build execution DAG for currently supported pipeline modes."""
    spec = resolve_pipeline_mode_spec(mode)
    nodes = spec.build_nodes(embedding_fusion_enabled)
    return StageGraph(mode=mode, nodes=nodes)
