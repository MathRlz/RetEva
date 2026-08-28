"""Runtime delivery of feature-node params (audit 2026-07 #4/#5/#7; R3-P3 resolution).

The loader folds a feature node's params into its sub-config, but BRANCH overrides survive
on the node and reach the handler through the **resolved per-node config**
(``evaluation/node_config.py``, computed once at run setup — it replaced the run-time
``node_overlay`` allowlist). These tests pin that path: overlaid keys win, unset keys
inherit the global (incl. the LLM backend), an explicit ``enabled: false`` beats the
presence-derived force, and a globally-absent feature stays absent.
"""
from types import MethodType, SimpleNamespace

from evaluator.config.query_correction import QueryCorrectionConfig
from evaluator.evaluation.executor.state import RunState
from evaluator.evaluation.node_config import resolve_node_config


def _state_with(node_id, resolved, **attrs):
    s = SimpleNamespace(
        current_node=SimpleNamespace(id=node_id),
        node_configs={node_id: resolved},
        **attrs,
    )
    s.resolved_config = MethodType(RunState.resolved_config, s)
    return s


def test_overlay_param_reaches_config_and_rest_inherits():
    base = QueryCorrectionConfig(enabled=True, method="rule", model="llama3.1",
                                 api_base="http://local:1234", temperature=0.3)
    cfg = resolve_node_config(base, {"op": "correct", "axis": "query", "method": "llm"})
    assert cfg.method == "llm"                    # node param applied
    assert cfg.model == "llama3.1"                # LLM backend inherited, not reset (#4)
    assert cfg.api_base == "http://local:1234"
    assert cfg.temperature == 0.3


def test_overlay_enabled_false_beats_force():
    base = QueryCorrectionConfig(enabled=True)
    cfg = resolve_node_config(base, {"enabled": False}, force_enabled=True)
    assert cfg.enabled is False                   # per-branch disable wins over force (#5)


def test_overlay_no_params_falls_back_to_global():
    base = QueryCorrectionConfig(enabled=True, method="kb")
    # discriminators only — they select behavior, they are never config values
    cfg = resolve_node_config(base, {"op": "correct", "axis": "query"})
    assert cfg is base                            # untouched global object


def test_globally_absent_feature_cannot_be_resurrected_by_params():
    assert resolve_node_config(None, {"enabled": True, "method": "rule"}) is None


def test_params_are_cast_to_the_field_type():
    # YAML/form values arrive as strings; casts come from the dataclass field types.
    base = QueryCorrectionConfig(enabled=False, kb_max_distance=1)
    cfg = resolve_node_config(base, {"enabled": "true", "kb_max_distance": "3"})
    assert cfg.enabled is True and cfg.kb_max_distance == 3


def test_unknown_param_is_ignored():
    base = QueryCorrectionConfig(method="rule")
    cfg = resolve_node_config(base, {"not_a_field": "x"})
    assert cfg is base


def test_correction_handler_reads_the_resolved_config():
    # The real handler helper (#4): a node overriding only `method` keeps the global backend.
    from evaluator.evaluation.handlers.query import _node_correction_config

    base = QueryCorrectionConfig(enabled=True, method="rule", model="mistral:7b",
                                 use_local_server=True,
                                 local_server_url="http://localhost:11434")
    s = _state_with(
        "query_correction", resolve_node_config(base, {"method": "llm"}),
        query_correction_config=base,
    )
    cfg = _node_correction_config(s)
    assert cfg.method == "llm"
    assert cfg.model == "mistral:7b"              # was reset to gpt-4o-mini before the fix
    assert cfg.use_local_server is True
    assert cfg.local_server_url == "http://localhost:11434"


def test_unresolved_node_falls_back_to_the_global():
    # A direct-call path that skipped resolution still gets the global config.
    from evaluator.evaluation.handlers.query import _node_correction_config

    base = QueryCorrectionConfig(enabled=True, method="kb")
    s = SimpleNamespace(
        current_node=SimpleNamespace(id="other"), node_configs={},
        query_correction_config=base,
    )
    s.resolved_config = MethodType(RunState.resolved_config, s)
    assert _node_correction_config(s) is base


def test_node_enabled_false_survives_the_load_fold():
    # Load-time half of #5: an explicit {enabled: false} on a feature node wins over the
    # presence-derived enabled:True (the graph is the spec).
    from evaluator.config.evaluation import EvaluationConfig
    from evaluator.config.graph_config import build_evaluation_config_kwargs

    from tests.graph_test_helpers import explicit_graph

    cfg = EvaluationConfig.from_dict(build_evaluation_config_kwargs({
        "graph": explicit_graph([
            "dataset_source", "asr",
            {"id": "query_correction", "type": "query_correction",
             "params": {"enabled": False, "method": "rule"}},
        ]),
    }), validate=False)
    assert cfg.query_correction.enabled is False
    assert cfg.query_correction.method == "rule"
