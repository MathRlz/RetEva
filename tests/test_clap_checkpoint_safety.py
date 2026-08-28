"""CLAP checkpoint load is safe-by-default (audit A1/F1)."""

import torch

from evaluator.models.a2e.clap_style import load_checkpoint_with_remap


class _Custom:  # a picklable non-tensor object → forces weights_only=False
    pass


def test_safe_tensor_checkpoint_loads_without_flag(tmp_path, monkeypatch):
    monkeypatch.delenv("EVALUATOR_ALLOW_UNSAFE_CHECKPOINTS", raising=False)
    p = tmp_path / "safe.pt"
    torch.save({"w": torch.zeros(3)}, p)
    ckpt = load_checkpoint_with_remap(str(p))
    assert "w" in ckpt


def test_unsafe_checkpoint_blocked_without_opt_in(tmp_path, monkeypatch):
    monkeypatch.delenv("EVALUATOR_ALLOW_UNSAFE_CHECKPOINTS", raising=False)
    import __main__

    __main__._Custom = _Custom  # picklable via the main module
    p = tmp_path / "unsafe.pt"
    torch.save({"obj": _Custom()}, p)
    import pytest

    with pytest.raises(RuntimeError, match="EVALUATOR_ALLOW_UNSAFE_CHECKPOINTS"):
        load_checkpoint_with_remap(str(p))


def test_unsafe_checkpoint_loads_with_opt_in(tmp_path, monkeypatch):
    import __main__

    __main__._Custom = _Custom
    monkeypatch.setenv("EVALUATOR_ALLOW_UNSAFE_CHECKPOINTS", "1")
    p = tmp_path / "unsafe.pt"
    torch.save({"obj": _Custom()}, p)
    ckpt = load_checkpoint_with_remap(str(p))
    assert "obj" in ckpt
