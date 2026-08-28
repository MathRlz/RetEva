"""R9 pins: handler registration is explicit-import driven and double-import safe.

The stage registry raises on a REAL name collision but tolerates the same handler
re-registered (a module imported under a second sys.modules name re-executes its
decorators — previously a hard crash).
"""

import importlib.util
import sys

import pytest

from evaluator.evaluation import stage_registry


def test_double_import_of_a_handler_module_is_idempotent():
    import evaluator.evaluation.handlers.asr  # ensure first registration happened

    spec = importlib.util.find_spec("evaluator.evaluation.handlers.asr")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_asr_handlers_second_copy"] = mod
    try:
        spec.loader.exec_module(mod)  # re-fires @register_stage_handler — must not raise
    finally:
        del sys.modules["_asr_handlers_second_copy"]
    assert stage_registry.get_stage_spec("asr") is not None


def test_conflicting_handler_registration_still_raises():
    with pytest.raises(ValueError, match="already registered"):
        @stage_registry.register_stage_handler("asr")
        def _imposter(s):  # a different function under an existing stage name
            pass
