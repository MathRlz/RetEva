"""Roadmap 1a — quantization knob: config resolution + opt-in factory application."""

import logging

from evaluator.config.model import ModelConfig
from evaluator.models.factory import _quantization_into_params
from tests.graph_test_helpers import mode_graph


def test_quantization_for_per_family_wins_over_global():
    c = ModelConfig(quantization="int8", asr_quantization="4bit")
    assert c.quantization_for("asr") == "4bit"       # per-family override
    assert c.quantization_for("text_emb") == "int8"  # falls back to global
    assert ModelConfig().quantization_for("asr") is None  # default off


class _Supports:
    def __init__(self, name, quantization=None):
        pass


class _Kwargs:
    def __init__(self, name, **kw):
        pass


class _NoSupport:
    def __init__(self, name):
        pass


def test_folds_quantization_only_when_model_supports_it(caplog):
    assert _quantization_into_params("x", _Supports, "int8", {}) == {"quantization": "int8"}
    assert _quantization_into_params("x", _Kwargs, "int8", {}) == {"quantization": "int8"}
    # unsupported → warn + unchanged (not a cryptic constructor error)
    with caplog.at_level(logging.WARNING, logger="evaluator.models.factory"):
        out = _quantization_into_params("x", _NoSupport, "int8", {"a": 1})
    assert out == {"a": 1}
    # None → no-op regardless of support
    assert _quantization_into_params("x", _NoSupport, None, {"a": 1}) == {"a": 1}


def test_node_centric_quantization_maps_to_legacy_field():
    from evaluator.config.graph_config import build_evaluation_config_kwargs

    legacy = build_evaluation_config_kwargs({
        "graph": mode_graph("asr_text_retrieval"),
        "nodes": {"asr": {"model": "whisper", "quantization": "int8"}},
    })
    assert legacy["model"]["asr_quantization"] == "int8"
