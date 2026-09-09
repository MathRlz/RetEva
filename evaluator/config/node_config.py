"""Per-node effective feature configs, resolved once before execution.

The allowlist IS each feature config's dataclass field set (casts come from the field
types), so it can't drift from the dataclass. The graph is walked once at run setup and
each feature node gets its own frozen config instance, read via ``RunState.resolved_config()``.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass, replace
from typing import Any, Dict, Optional

from ..logging_config import get_logger

logger = get_logger(__name__)

#: Operator-alias discriminator fields: they pick WHICH behavior a node has, so they must
#: never be overlaid onto a feature config even when the names happen to collide.
#: NOT the full `operators._DISCRIMINATORS` union — that includes ``method``, which IS a
#: real field of the correction / answer-gen / query-opt configs and must overlay.
DISCRIMINATOR_FIELDS = frozenset(
    {"op", "axis", "level", "family", "trace", "target", "union", "modality"}
)

#: node_kind → param names that must NOT overlay that kind's feature config, because the name
#: collides across namespaces. `query_refine`'s ``method`` picks a REFINE STRATEGY
#: (models/retrieval/query/optimization.py:_REFINE_STRATEGIES — rewrite_with_context /
#: relevance_feedback / self_rag_critique), while ``QueryOptimizationConfig.method`` is the
#: rewrite/hyde/decompose/multi_query family. Overlaying one onto the other made every refine
#: strategy name a hard ValueError, i.e. the node could not be used at all.
KIND_KEEP = {"query_refine": frozenset({"method"})}

#: node_kind → the RunState attribute holding that feature's global config. Nodes whose
#: presence implies the feature carry ``force_enabled``.
_FEATURE_BASES = {
    "query_correction": ("query_correction_config", False),
    "query_optimization": ("query_opt_config", False),
    "query_refine": ("query_opt_config", False),
    "multi_query_retrieval": ("query_opt_config", False),
    "answer_gen": ("answer_gen_config", True),
    "answer_judge": ("judge_config", True),
    "fusion": ("embedding_fusion_config", True),
    "tts": ("_config.audio_synthesis", True),
    # NB: the real EvaluationConfig field is `augmentation`, not `audio_augmentation` — the
    # wrong name here would make this global unreachable (getattr always None).
    "augment_audio": ("_config.augmentation", True),
}


def _cast(value: Any, field_type: Any) -> Any:
    """Coerce a param (YAML/form values arrive as strings) to the dataclass field's type;
    only scalar types are handled (containers/enums/unannotated pass through). ``field_type``
    is often a plain string (``from __future__ import annotations`` leaves it unevaluated),
    so both spellings are handled."""
    name = field_type if isinstance(field_type, str) else getattr(field_type, "__name__", "")
    if name == "bool":
        if isinstance(value, str):
            return value.strip().lower() not in ("false", "0", "no", "")
        return bool(value)
    if name == "int":
        return int(value)
    if name == "float":
        return float(value)
    if name == "str":
        return str(value)
    return value


def resolve_node_config(
    base: Any, params: Optional[dict], *, force_enabled: bool = False,
    keep_on_node: frozenset = frozenset(),
) -> Any:
    """``base`` with this node's ``params`` overlaid — the single resolution point.

    The overlay keys are exactly ``base``'s dataclass fields minus the discriminators, and
    each value is cast to its field's declared type. ``None`` base (feature disabled
    globally) returns ``None`` unchanged."""
    if base is None or not is_dataclass(base):
        return base
    types = {f.name: f.type for f in fields(base)}
    overlay: Dict[str, Any] = {}
    for key, value in (params or {}).items():
        if (key in DISCRIMINATOR_FIELDS or key in keep_on_node
                or key not in types or value in (None, "")):
            continue
        try:
            overlay[key] = _cast(value, types[key])
        except (TypeError, ValueError):
            logger.warning(
                "node param %s=%r is not a valid %s — keeping the inherited value",
                key, value, types[key],
            )
    if force_enabled and "enabled" in types:
        overlay.setdefault("enabled", True)
    return replace(base, **overlay) if overlay else base


def resolve_graph_node_configs(state: Any, stage_graph: Any) -> Dict[str, Any]:
    """``{node_id: resolved feature config}`` for every feature node in the graph.

    Called once from the executor setup (the graph and the global configs are both in hand
    there), so a handler's config is a plain lookup — no run-time overlay machinery."""
    from ..pipeline.graph.operators import node_kind

    resolved: Dict[str, Any] = {}
    for node in getattr(stage_graph, "nodes", ()):
        kind = node_kind(node.stage, node.params)
        entry = _FEATURE_BASES.get(kind)
        if entry is None:
            continue
        attr, force = entry
        if attr.startswith("_config."):
            cfg = getattr(state, "config", None)
            base = getattr(cfg, attr.split(".", 1)[1], None) if cfg is not None else None
        else:
            base = getattr(state, attr, None)
        resolved[node.id] = resolve_node_config(
            base, node.params, force_enabled=force,
            keep_on_node=KIND_KEEP.get(kind, frozenset()),
        )
    return resolved
