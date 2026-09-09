"""Public introspection surface for UI / form builders.

The **supported boundary**: consumers import from here instead of reaching into
``graph.registry`` privates, so internals can be refactored as long as this surface holds.
"""

from __future__ import annotations

from .graph.display import display_label
from .graph.operators import expand_alias, node_kind, operator_discriminators
from .graph.artifacts import STRUCTURAL_ARTIFACTS
from .graph.registry import _resolve as resolve_field
from .graph.registry import is_structural
from .graph.registry import node_category, node_domain, node_model_field
from .graph import effective_inputs, effective_outputs
from .graph import dataset_columns, display_artifact_names, get_stage_node_def

__all__ = [
    "resolve_field",
    "is_structural",
    "effective_inputs",
    "effective_outputs",
    "node_category",
    "node_domain",
    "node_model_field",
    "node_kind",
    "expand_alias",
    "operator_discriminators",
    "display_label",
    "display_artifact_names",
    "dataset_columns",
    "get_stage_node_def",
    "STRUCTURAL_ARTIFACTS",
]
