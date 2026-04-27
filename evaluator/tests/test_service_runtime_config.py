"""Tests for service runtime policy configuration."""

import pytest

from evaluator.config import EvaluationConfig, ServiceRuntimeConfig


def test_service_runtime_defaults():
    cfg = ServiceRuntimeConfig()
    assert cfg.startup_mode == "lazy"
    assert cfg.offload_policy == "on_finish"


def test_service_runtime_rejects_invalid_values():
    with pytest.raises(ValueError, match="startup_mode"):
        ServiceRuntimeConfig(startup_mode="invalid")
    with pytest.raises(ValueError, match="offload_policy"):
        ServiceRuntimeConfig(offload_policy="invalid")


def test_evaluation_config_from_dict_parses_service_runtime():
    config = EvaluationConfig.from_dict(
        {
            "service_runtime": {
                "startup_mode": "eager",
                "offload_policy": "never",
            }
        },
        validate=False,
    )
    assert config.service_runtime.startup_mode == "eager"
    assert config.service_runtime.offload_policy == "never"
