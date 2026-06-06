"""File-based evaluation presets.

Presets are the YAML configs in the repo ``configs/`` directory: any ``*.yaml`` there
that parses as a valid :class:`EvaluationConfig` is offered as a preset, named by its
file stem. The hardcoded preset dicts were removed so the runnable example configs and
the preset list are a single source of truth.

The configs directory is resolved relative to the repo by default, overridable with the
``EVALUATOR_CONFIGS_DIR`` environment variable.
"""

import os
from pathlib import Path
from typing import Any, Dict, List

import yaml

# ``configs/`` at the repo root (this file is ``evaluator/config/model_presets.py``),
# overridable for non-standard layouts.
CONFIGS_DIR = Path(
    os.environ.get("EVALUATOR_CONFIGS_DIR")
    or (Path(__file__).resolve().parents[2] / "configs")
)


def _preset_paths() -> Dict[str, Path]:
    """Map preset name (file stem) -> path for every ``configs/*.yaml``."""
    if not CONFIGS_DIR.is_dir():
        return {}
    return {path.stem: path for path in sorted(CONFIGS_DIR.glob("*.yaml"))}


def _is_valid_preset(path: Path) -> bool:
    """A preset is any config that builds an EvaluationConfig (structure only).

    ``validate=False`` skips the run-time checks (file existence, registries) so a
    config still counts as a preset even when its dataset/corpus files aren't present;
    grid/ablation YAMLs that aren't a single EvaluationConfig fail here and are excluded.
    """
    from .evaluation import EvaluationConfig
    try:
        EvaluationConfig.from_yaml(str(path), validate=False)
        return True
    except Exception:
        return False


def _apply_auto_devices(preset: Dict[str, Any]) -> None:
    """Override the preset's model device fields based on available GPUs."""
    from . import get_available_gpu_count

    model = preset.setdefault("model", {})
    gpu_count = get_available_gpu_count()
    if gpu_count == 0:
        devices = {"asr_device": "cpu", "text_emb_device": "cpu", "audio_emb_device": "cpu"}
    elif gpu_count == 1:
        devices = {"asr_device": "cuda:0", "text_emb_device": "cuda:0", "audio_emb_device": "cuda:0"}
    else:
        devices = {"asr_device": "cuda:0", "text_emb_device": "cuda:1", "audio_emb_device": "cuda:0"}
    model.update(devices)


def list_presets() -> List[str]:
    """Return preset names: ``configs/*.yaml`` that parse as a valid EvaluationConfig."""
    return [name for name, path in _preset_paths().items() if _is_valid_preset(path)]


def get_preset(name: str, auto_devices: bool = True) -> Dict[str, Any]:
    """Load preset ``name`` (``configs/<name>.yaml``) as a config dict.

    Args:
        name: Preset name — the YAML file stem under ``configs/``.
        auto_devices: If True (default), override model device fields for the
            available hardware. Set False to keep the file's device assignments.

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
        preset: Dict[str, Any] = yaml.safe_load(handle) or {}
    if auto_devices:
        _apply_auto_devices(preset)
    return preset
