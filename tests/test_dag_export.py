"""The DOT DAG exporter emits well-formed, category-colored, artifact-labeled output."""

from evaluator.config.evaluation import EvaluationConfig
from evaluator.pipeline.graph.export import graph_to_dot
from evaluator.pipeline.graph.build import build_graph_for_config
from tests.fixtures_paths import E2E_MINI


def test_dot_export_structure():
    cfg = EvaluationConfig.from_yaml(E2E_MINI)
    dot = graph_to_dot(build_graph_for_config(cfg), title="t")
    assert dot.startswith("digraph evaluator_dag {")
    assert dot.rstrip().endswith("}")
    # every node declared, source colored blue, a data-flow edge carries its artifact
    assert '"dataset_source" [label=' in dot
    assert 'fillcolor="#dbeafe"' in dot
    assert '"asr" -> ' in dot and '[label="query_text"]' in dot
    # quoted ids stay balanced
    assert dot.count("{") == dot.count("}")
