"""DAG-lite stage graph for current pipeline modes (split into a package).

Two registries drive graph construction:

* ``register_stage_node`` — declares a node *type* (its name + the config field of the
  model it runs, if any). This is the structural twin of the executable handler registry
  in ``evaluation/stage_registry.py``: a node declares itself once and introspection
  (model lifecycle, required-field derivation) reads it, instead of a hardcoded per-mode
  map.
* ``build_graph_from_spec`` — the one way a graph is built: from an ordered list of node
  ids, each node is auto-wired to the earlier nodes that produce its required / present
  optional input artifacts (``edges`` adds ordering not implied by data). Named modes
  (``PIPELINE_MODE_SPECS``) are just ordered node-id lists fed to this engine.

This ``__init__`` re-exports the full public (and historically-imported private) surface
so ``from evaluator.pipeline.graph import X`` and the legacy ``evaluator.pipeline.stage_graph``
shim both keep working unchanged.
"""

from .branches import build_branched_graph, expand_branches  # noqa: F401
from .cse import (  # noqa: F401  (re-export surface)
    _freeze_params,
    _topo_order,
    collapse_common_subexpressions,
)
from .modes import (  # noqa: F401  (re-export surface)
    PIPELINE_MODE_SPECS,
    PipelineModeSpec,
    _config_rerank_enabled,
    _make_spec,
    _mode_node_ids,
    _required_model_fields,
    build_graph_for_config,
    build_stage_graph,
    list_pipeline_mode_specs,
    resolve_pipeline_mode_spec,
)
from .registry import (  # noqa: F401  (re-export surface)
    ARTIFACT_CORPUS,
    ARTIFACT_GENERATED_ANSWERS,
    ARTIFACT_METRICS,
    ARTIFACT_QUERY_AUDIO,
    ARTIFACT_QUERY_TEXT,
    ARTIFACT_QUERY_VECTORS,
    ARTIFACT_RELEVANT_DOCS,
    ARTIFACT_RETRIEVED,
    ARTIFACT_SHORT_ANSWERS,
    ARTIFACT_VECTOR_INDEX,
    DATASET_ROLE_BOTH,
    DATASET_ROLE_CORPUS,
    DATASET_ROLE_QUESTIONS,
    SOURCE_ARTIFACTS,
    _DATASET_SOURCE_ROLE_OUTPUTS,
    _effective_outputs,
    _NODE_REGISTRY,
    StageGraph,
    StageNode,
    StageNodeDef,
    get_stage_node_def,
    register_stage_node,
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
    "PipelineModeSpec",
    # Registry
    "register_stage_node",
    "get_stage_node_def",
    "validate_graph_artifacts",
    # Wiring
    "build_graph_from_spec",
    # CSE
    "collapse_common_subexpressions",
    # Branches
    "expand_branches",
    "build_branched_graph",
    # Modes
    "build_stage_graph",
    "build_graph_for_config",
    "list_pipeline_mode_specs",
    "resolve_pipeline_mode_spec",
    "PIPELINE_MODE_SPECS",
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
    "ARTIFACT_RELEVANT_DOCS",
    "ARTIFACT_RETRIEVED",
    "ARTIFACT_SHORT_ANSWERS",
    "ARTIFACT_VECTOR_INDEX",
]
