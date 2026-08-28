"""The hybrid-fusion registry (OCP): built-ins are registered, `fuse_hybrid_results` dispatches via
the registry, a plugin strategy is reachable, an unknown method errors with the list, and both
config validation and the builder node form read the registry (so a new strategy auto-appears)."""
import pytest

from evaluator.models.retrieval import (
    FUSION_REGISTRY,
    fuse_hybrid_results,
    list_fusions,
    register_fusion,
)


def _pick_first_fusion(dense, sparse, *, dense_weight, top_k, rrf_k):
    """Trivial fusion fn: return the first `top_k` dense results (enough to prove dispatch)."""
    return list(dense)[:top_k]


def test_builtins_registered():
    assert {"weighted", "rrf", "max_score"} <= set(list_fusions())


def test_unknown_method_errors_with_available_list():
    with pytest.raises(ValueError, match="Unsupported hybrid fusion method"):
        fuse_hybrid_results("nope", [], [], dense_weight=0.5, top_k=5, rrf_k=60)


def test_plugin_strategy_is_dispatched():
    register_fusion("test_pick_first", _pick_first_fusion)
    try:
        out = fuse_hybrid_results(
            "test_pick_first", [("a", 1.0), ("b", 0.5)], [("c", 0.9)],
            dense_weight=0.5, top_k=1, rrf_k=60,
        )
        assert out == [("a", 1.0)]
        assert "test_pick_first" in list_fusions()
    finally:
        FUSION_REGISTRY.pop("test_pick_first", None)


def test_config_validation_accepts_a_registered_fusion():
    from evaluator.models.retrieval.strategy import CoreRetrievalConfig

    register_fusion("test_val_fusion", _pick_first_fusion)
    try:
        CoreRetrievalConfig(
            mode="hybrid", hybrid_fusion_method="test_val_fusion"
        ).validate()  # must not raise
    finally:
        FUSION_REGISTRY.pop("test_val_fusion", None)


def test_builder_combine_form_lists_registered_fusions():
    from evaluator.webapi.form_builder import resolve_node_form

    register_fusion("test_ui_fusion", _pick_first_fusion)
    try:
        form = resolve_node_form("combine", {"level": "result"})
        method = next(p for p in form["node_params"] if p["key"] == "method")
        assert "test_ui_fusion" in method["choices"]
    finally:
        FUSION_REGISTRY.pop("test_ui_fusion", None)


# ── multi-query combine-strategy registry (the parallel, separate from hybrid fusion) ──


def test_combine_strategy_builtins_and_plugin():
    from evaluator.models.retrieval.query.optimization import (
        combine_retrieval_results,
        list_combine_strategies,
        register_combine_strategy,
        _COMBINE_REGISTRY,
    )

    assert {"rrf", "weighted", "union", "intersection"} <= set(list_combine_strategies())
    register_combine_strategy(
        "test_pick_last", lambda rl, *, k, rrf_k, weights: rl[-1][:k]
    )
    try:
        out = combine_retrieval_results(
            [[("a", 1.0)], [("b", 0.5)]], strategy="test_pick_last", k=1
        )
        assert out == [("b", 0.5)]
        assert "test_pick_last" in list_combine_strategies()
    finally:
        _COMBINE_REGISTRY.pop("test_pick_last", None)


def test_combine_strategy_unknown_errors():
    from evaluator.models.retrieval.query.optimization import combine_retrieval_results

    with pytest.raises(ValueError, match="Unknown combination strategy"):
        combine_retrieval_results([[("a", 1.0)], [("b", 0.5)]], strategy="nope")


def test_builder_search_form_lists_combine_strategies():
    from evaluator.webapi.form_builder import resolve_node_form
    from evaluator.models.retrieval.query.optimization import (
        register_combine_strategy,
        _COMBINE_REGISTRY,
    )

    register_combine_strategy(
        "test_ui_combine", lambda rl, *, k, rrf_k, weights: rl[0][:k]
    )
    try:
        form = resolve_node_form("search", {"method": "multi_query"})
        strat = next(p for p in form["node_params"] if p["key"] == "combine_strategy")
        assert "test_ui_combine" in strat["choices"]
    finally:
        _COMBINE_REGISTRY.pop("test_ui_combine", None)
