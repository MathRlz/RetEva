"""File-based evaluation presets.

Presets are the YAML configs in the repo ``configs/`` directory; any ``*.yaml`` that
parses as a valid :class:`EvaluationConfig` is offered as a preset, named by its file stem.
"""

import os
from pathlib import Path
from typing import Any, Dict, List

import yaml
from .base import get_available_gpu_count
from .graph_config import build_evaluation_config_kwargs


def _configs_dir() -> Path:
    """Resolve the presets directory (lazy — env var honored per call): ``EVALUATOR_CONFIGS_DIR``
    → ``configs/`` at the repo root → ``configs/`` under the cwd (first hit wins)."""
    env = os.environ.get("EVALUATOR_CONFIGS_DIR")
    if env:
        return Path(env)
    repo = Path(__file__).resolve().parents[2] / "configs"
    if repo.is_dir():
        return repo
    return Path.cwd() / "configs"


def _preset_paths() -> Dict[str, Path]:
    """Map preset name (file stem) -> path for every ``configs/*.yaml``."""
    configs_dir = _configs_dir()
    if not configs_dir.is_dir():
        return {}
    return {path.stem: path for path in sorted(configs_dir.glob("*.yaml"))}


def _is_valid_preset(path: Path) -> bool:
    """A preset is any config that builds an EvaluationConfig (structure only —
    ``validate=False`` skips run-time checks like file existence); grid/ablation YAMLs
    that aren't a single EvaluationConfig fail here and are excluded."""
    from .evaluation import EvaluationConfig  # cycle: evaluation builds via loading

    try:
        EvaluationConfig.from_yaml(str(path), validate=False)
        return True
    except Exception:
        return False


def _apply_auto_devices(preset: Dict[str, Any]) -> None:
    """Override the preset's model device fields based on available GPUs."""

    model = preset.setdefault("model", {})
    gpu_count = get_available_gpu_count()
    if gpu_count == 0:
        devices = {
            "asr_device": "cpu",
            "text_emb_device": "cpu",
            "audio_emb_device": "cpu",
        }
    elif gpu_count == 1:
        devices = {
            "asr_device": "cuda:0",
            "text_emb_device": "cuda:0",
            "audio_emb_device": "cuda:0",
        }
    else:
        devices = {
            "asr_device": "cuda:0",
            "text_emb_device": "cuda:1",
            "audio_emb_device": "cuda:0",
        }
    model.update(devices)


def list_presets() -> List[str]:
    """Return preset names: ``configs/*.yaml`` that parse as a valid EvaluationConfig."""
    return [name for name, path in _preset_paths().items() if _is_valid_preset(path)]


def get_preset(name: str, auto_devices: bool = True) -> Dict[str, Any]:
    """Load preset ``name`` (``configs/<name>.yaml``) as a config dict.

    Args:
        auto_devices: True (default) overrides model device fields for the available
            hardware; False keeps the file's device assignments.

    Returns:
        The parsed config dict (a fresh object each call).

    Raises:
        ValueError: If no valid preset with that name exists.
    """
    path = _preset_paths().get(name)
    if path is None or not _is_valid_preset(path):
        raise ValueError(
            f"Unknown preset '{name}'. Available presets: {', '.join(list_presets())}"
        )

    with open(path, "r", encoding="utf-8") as handle:
        raw: Dict[str, Any] = yaml.safe_load(handle) or {}
    # Presets are node-centric YAML; translate to the from_dict kwargs the rest of the
    # pipeline (and _apply_auto_devices' model.* device fields) expects.
    preset = build_evaluation_config_kwargs(raw)
    if auto_devices:
        _apply_auto_devices(preset)
    return preset
