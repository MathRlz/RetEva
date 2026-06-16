"""Public introspection surface for UI / form builders.

The webapi builds node catalogues, graph previews and field-aware forms by inspecting the graph
registry. This module is the **supported boundary** for that: consumers import from here instead
of reaching into ``graph.registry`` / ``stage_graph`` privates (``_resolve`` / ``_effective_*``),
so the internals can be refactored as long as this surface holds. Names that are already public
are re-exported too, so a UI module has one import site.
"""

from __future__ import annotations

from .graph.display import display_label
from .graph.operators import expand_alias, node_kind, operator_discriminators
from .graph.registry import _resolve as resolve_field
from .graph.registry import node_category, node_domain, node_model_field
from .stage_graph import _effective_inputs as effective_inputs
from .stage_graph import _effective_outputs as effective_outputs
from .stage_graph import dataset_columns, display_artifact_names, get_stage_node_def

__all__ = [
    "resolve_field",
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
]
