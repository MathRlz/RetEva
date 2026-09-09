"""The config → evaluation registry-validator hook stays wired.

`config/` sits below the metric, model and query registries, so the name checks that consult
them live in ``evaluation/validation.py`` and plug themselves into
``config.validation.EXTRA_VALIDATORS`` on import. Nothing in `config/` references them, and the
import that installs them rides on ``evaluator/__init__`` pulling in ``evaluator.evaluation``:
if that chain ever stops running, ``config.validate()`` quietly stops checking metric, model,
corrector and combine-strategy names, and every other test still passes. These pin it.
"""

import os
import subprocess
import sys

import pytest

from evaluator.config.evaluation import EvaluationConfig
from evaluator.config.graph_config import build_evaluation_config_kwargs
from evaluator.config.validation import EXTRA_VALIDATORS
from evaluator.errors import ConfigurationError
from evaluator.evaluation.validation import collect_registry_problems
from tests.graph_test_helpers import mode_graph

_OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
}


def _config_dict(**extra):
    """A minimal explicit-graph config dict, plus whatever the case under test overrides."""
    base = {
        "graph": mode_graph("asr_text_retrieval"),
        "nodes": {"asr": {"model": "whisper"}, "text_embedding": {"model": "labse"}},
    }
    base.update(extra)
    return build_evaluation_config_kwargs(base)


def test_registry_checks_are_wired_into_config_validate():
    assert collect_registry_problems in EXTRA_VALIDATORS

    # A user reaching for the config alone must get the same checks: importing only
    # evaluator.config has to be enough for the hook to be installed, which it is because
    # evaluator/__init__ pulls public_api -> services -> evaluator.evaluation.
    env = {**os.environ, **_OFFLINE_ENV}
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import evaluator.config as c; "
            "from evaluator.config.validation import EXTRA_VALIDATORS; "
            "print(len(EXTRA_VALIDATORS))",
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    assert proc.returncode == 0, proc.stderr
    assert int(proc.stdout.strip().splitlines()[-1]) >= 1, proc.stdout


def test_clean_config_validates():
    EvaluationConfig.from_dict(_config_dict(), validate=True)  # must not raise


def test_bad_metric_name_is_rejected_by_from_dict():
    with pytest.raises(ConfigurationError) as exc:
        EvaluationConfig.from_dict(_config_dict(metrics=["nope"]), validate=True)
    msg = str(exc.value)
    assert "nope" in msg and "Registered:" in msg


def test_bad_corrector_method_is_rejected_by_validate():
    # The corrector-name check is a registry lookup, not a dataclass-level constraint, so it
    # surfaces as a ConfigurationError from validate() (pre-flight), not a ValueError from
    # QueryCorrectionConfig.__post_init__.
    config = EvaluationConfig.from_dict(
        _config_dict(query_correction={"method": "typo"}), validate=False
    )
    with pytest.raises(ConfigurationError, match="query_correction.method must be one of"):
        config.validate()
