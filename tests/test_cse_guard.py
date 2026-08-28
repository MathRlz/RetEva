"""A2 pins (CRITIQUE.md): CSE must never over-share.

Two branch nodes collapse iff their canonical key matches; the key drops params equal to a
registered ``param_defaults`` constant (S7). That drop is sound ONLY when the handler's
absent-param fallback is that same constant in every config. A param whose absent value is
config-inherited (the index node's ``store`` inherits the global ``vector_db.type``) must
not be declared — an explicit value and the inherited global would collapse into one node
and one branch would silently run with the other's backend.
"""

from evaluator.pipeline.graph.cse import collapse_common_subexpressions
from evaluator.pipeline.graph.registry import _NODE_REGISTRY, StageNode

# Every registered param_defaults entry needs a row here, added only after checking the
# handler's absent-param fallback is this exact constant regardless of config.
SOUND_CSE_DEFAULTS: dict = {
    # stage -> {param: constant}
}


def test_param_defaults_are_allowlisted_constant_fallbacks():
    declared = {
        stage: dict(node_def.param_defaults)
        for stage, node_def in _NODE_REGISTRY.items()
        if node_def.param_defaults
    }
    assert declared == SOUND_CSE_DEFAULTS, (
        "param_defaults changed. A CSE default collapses an explicit value with an omitted "
        "one — sound only when the handler's absent-param fallback is that constant in every "
        "config (never config-inherited, like index's `store`). Verify, then update "
        "SOUND_CSE_DEFAULTS."
    )


def _twin(node_id: str, params) -> StageNode:
    return StageNode(id=node_id, stage="index", depends_on=(), bindings=(), params=params)


def test_explicit_store_does_not_collapse_with_inherited_store():
    # {} means "inherit the global vector_db.type"; an explicit inmemory is a different node.
    kept = collapse_common_subexpressions(
        (_twin("vector_db@a", None), _twin("vector_db@b", {"store": "inmemory"}))
    )
    assert len(kept) == 2


def test_identical_twins_still_collapse():
    kept = collapse_common_subexpressions(
        (_twin("vector_db@a", {"store": "faiss"}), _twin("vector_db@b", {"store": "faiss"}))
    )
    assert len(kept) == 1


def test_explicit_none_collapses_with_omitted():
    # Documented semantic: None == absent (handlers read params via .get, so an explicit
    # YAML `key:` (None) cannot behave differently from an omitted key).
    kept = collapse_common_subexpressions(
        (_twin("vector_db@a", {"store": None}), _twin("vector_db@b", None))
    )
    assert len(kept) == 1
