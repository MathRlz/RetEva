"""DAG-lite stage graph (split into a package).

This ``__init__`` re-exports the package's public surface, so
``from evaluator.pipeline.graph import X`` works for every submodule.
"""

from .display import display_label  # noqa: F401
from .operators import (  # noqa: F401  (re-export surface)
    ALIASES,
    OPERATORS,
    expand_alias,
    is_operator,
    node_kind,
)
from .cse import (  # noqa: F401  (re-export surface)
    _freeze_params,
    _topo_order,
    collapse_common_subexpressions,
)
from .build import (  # noqa: F401  (re-export surface)
    build_graph_for_config,
    build_run_graph,
    complete_structural_plumbing,
    graph_label,
    run_label,
    structural_specs,
)
from .registry import (  # noqa: F401  (re-export surface)
    ARTIFACT_CORPUS,
    ARTIFACT_GENERATED_ANSWERS,
    ARTIFACT_METRICS,
    ARTIFACT_QUERY_AUDIO,
    ARTIFACT_QUERY_TEXT,
    ARTIFACT_QUERY_VECTORS,
    ARTIFACT_AUDIO_QUERY_VECTORS,
    ARTIFACT_TEXT_QUERY_VECTORS,
    ARTIFACT_FUSED_QUERY_VECTORS,
    ARTIFACT_REFERENCE_TRANSCRIPTION,
    ARTIFACT_EMBEDDING_ALIGNMENT,
    ARTIFACT_RELEVANT_DOCS,
    ARTIFACT_RETRIEVED,
    ARTIFACT_SHORT_ANSWERS,
    ARTIFACT_CORPUS_VECTORS,
    ARTIFACT_VECTOR_INDEX,
    DATASET_ROLE_BOTH,
    DATASET_ROLE_CORPUS,
    DATASET_ROLE_QUESTIONS,
    SOURCE_ARTIFACTS,
    _DATASET_SOURCE_ROLE_OUTPUTS,
    effective_inputs,
    effective_outputs,
    dataset_columns,
    display_artifact_names,
    NODE_REGISTRY,
    OneOf,
    one_of,
    StageGraph,
    StageNode,
    StageNodeDef,
    get_stage_node_def,
    node_category,
    node_domain,
    node_model_field,
    register_stage_node,
    registered_stage_names,
    stages_in_category,
    validate_graph_artifacts,
)
from .wiring import (  # noqa: F401  (re-export surface)
    _normalize_spec_item,
    _wire_nodes,
    build_graph_from_spec,
)

__all__ = [
    # Data structures
    "StageNode",
    "StageNodeDef",
    "StageGraph",
    # Registry
    "NODE_REGISTRY",
    "register_stage_node",
    "registered_stage_names",
    "stages_in_category",
    "node_category",
    "node_domain",
    "node_model_field",
    "get_stage_node_def",
    "validate_graph_artifacts",
    "effective_inputs",
    "effective_outputs",
    # Operators (alias bijection)
    "ALIASES",
    "OPERATORS",
    "expand_alias",
    "is_operator",
    "node_kind",
    "OneOf",
    "one_of",
    "display_artifact_names",
    # Wiring
    "build_graph_from_spec",
    # CSE
    "collapse_common_subexpressions",
    # Display
    "display_label",
    # Build
    "build_graph_for_config",
    "build_run_graph",
    "complete_structural_plumbing",
    "graph_label",
    "run_label",
    "structural_specs",
    # Dataset roles
    "DATASET_ROLE_BOTH",
    "DATASET_ROLE_CORPUS",
    "DATASET_ROLE_QUESTIONS",
    # Artifact vocabulary
    "SOURCE_ARTIFACTS",
    "ARTIFACT_CORPUS",
    "ARTIFACT_GENERATED_ANSWERS",
    "ARTIFACT_METRICS",
    "ARTIFACT_QUERY_AUDIO",
    "ARTIFACT_QUERY_TEXT",
    "ARTIFACT_QUERY_VECTORS",
    "ARTIFACT_AUDIO_QUERY_VECTORS",
    "ARTIFACT_TEXT_QUERY_VECTORS",
    "ARTIFACT_FUSED_QUERY_VECTORS",
    "ARTIFACT_REFERENCE_TRANSCRIPTION",
    "ARTIFACT_EMBEDDING_ALIGNMENT",
    "ARTIFACT_RELEVANT_DOCS",
    "ARTIFACT_RETRIEVED",
    "ARTIFACT_SHORT_ANSWERS",
    "ARTIFACT_CORPUS_VECTORS",
    "ARTIFACT_VECTOR_INDEX",
]
