"""``EvaluationConfig.with_auto_devices()`` on a config that carries a ``device_pool``: the
pool-allocation branch is the only one that reaches ``evaluator.devices``, and a wrong relative
import there is invisible until a pooled config asks for its devices."""

import os

import pytest

from evaluator.config.evaluation import EvaluationConfig
from evaluator.devices import pool as pool_module

CONFIG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "configs",
    "admed_audioemb_noise_snr20_cmp.yaml",
)


@pytest.fixture(autouse=True)
def _no_leaked_active_pool(monkeypatch):
    # pool_from_config publishes the pool it builds as the run's active one; keep that off
    # the other tests' path.
    monkeypatch.setattr(pool_module, "_active_pool", None)


def test_with_auto_devices_allocates_from_the_pool():
    config = EvaluationConfig.from_yaml(CONFIG, validate=False)
    assert config.device_pool is not None

    configured = config.with_auto_devices()

    # cuda:N on a GPU box, "cpu" via the pool's fallback here — either way a real device
    # string, which only happens if devices.pool imported and allocated.
    for attr in ("asr_device", "text_emb_device", "audio_emb_device"):
        model_type = getattr(config.model, attr.replace("_device", "_model_type"))
        if model_type:
            assert getattr(configured.model, attr)
