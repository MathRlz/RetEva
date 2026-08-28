"""Smoke: every apm_tests config loads → builds its DAG → passes space validation.

No models are loaded; this guards config schema + graph wiring + the embedding-space
contract against regressions when configs or the translator change.
"""

from pathlib import Path

import pytest

from evaluator.config.evaluation import EvaluationConfig
from evaluator.pipeline import build_graph_for_config
from evaluator.evaluation.validation import validate_graph_embedding_spaces

_CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs" / "apm_tests"
_CONFIGS = sorted(str(p) for p in _CONFIG_ROOT.rglob("*.yaml"))


def test_configs_present():
    assert _CONFIGS, f"no apm_tests configs found under {_CONFIG_ROOT}"


@pytest.mark.parametrize("path", _CONFIGS, ids=lambda p: Path(p).name)
def test_config_loads_builds_and_validates(path):
    cfg = EvaluationConfig.from_yaml(path)
    graph = build_graph_for_config(cfg)
    assert graph.node_ids(), f"{path}: empty graph"
    validate_graph_embedding_spaces(graph, cfg)  # APM configs declare a shared embedding_space
