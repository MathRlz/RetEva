"""Declarative parameter sweeps (architecture-improvements §3).

A *sweep spec* names a base config + a set of axes; expanding it yields one
``EvaluationConfig`` per point in the cartesian product, all tagged with a shared
``experiment_group`` so the leaderboard can pivot/compare them as one experiment::

    name: k_sweep
    base: configs/e2e_pubmed_qa_small.yaml   # path, preset name, or an inline config dict
    axes:
      - path: vector_db.k
        values: [5, 10, 20]
      - path: vector_db.retrieval_mode
        values: [dense, hybrid]

``expand_sweep`` is pure (no models loaded), so it is cheap to preview/test; running the
expanded configs reuses the normal evaluation path.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from ..config.evaluation import EvaluationConfig
from ..errors import ConfigurationError
from .grid_search import GridSearch


def load_sweep_spec(path: str) -> Dict[str, Any]:
    """Load a sweep YAML/JSON spec into a dict (validated shape)."""
    import json

    if path.endswith((".yaml", ".yml")):
        import yaml

        with open(path, "r", encoding="utf-8") as fh:
            spec = yaml.safe_load(fh)
    else:
        with open(path, "r", encoding="utf-8") as fh:
            spec = json.load(fh)
    _validate_spec(spec)
    return spec


def _validate_spec(spec: Any) -> None:
    if not isinstance(spec, dict) or "base" not in spec or "axes" not in spec:
        raise ConfigurationError(
            "sweep spec must be a mapping with 'base' and 'axes' keys"
        )
    if not isinstance(spec["axes"], list) or not spec["axes"]:
        raise ConfigurationError("sweep 'axes' must be a non-empty list")
    for axis in spec["axes"]:
        if not isinstance(axis, dict) or "path" not in axis or "values" not in axis:
            raise ConfigurationError(
                "each sweep axis needs a 'path' and a 'values' list"
            )
        if not isinstance(axis["values"], list) or not axis["values"]:
            raise ConfigurationError(
                f"sweep axis '{axis.get('path')}' needs a non-empty 'values' list"
            )


def _load_base_config(base: Any) -> EvaluationConfig:
    """Resolve the sweep base into an EvaluationConfig (path / preset name / inline dict)."""
    if isinstance(base, dict):
        return EvaluationConfig.from_dict(base, validate=False)
    if isinstance(base, str):
        if base.endswith((".yaml", ".yml", ".json")):
            return EvaluationConfig.from_yaml(base)
        return EvaluationConfig.from_preset(base, validate=False)
    raise ConfigurationError(f"unsupported sweep base: {type(base).__name__}")


def combo_label(combo: Dict[str, Any]) -> str:
    """Compact, stable label for one axis combination, e.g. ``k=10,retrieval_mode=hybrid``."""
    return ",".join(f"{p.split('.')[-1]}={v}" for p, v in sorted(combo.items()))


def sweep_preview(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Expand a sweep spec to its combination list WITHOUT running anything (Roadmap 4a).

    The builder's sweep form posts axes and gets back the run group it would launch:
    ``{name, count, combos: [{label, experiment_name, combo}, …]}`` — so a researcher sees the
    full grid (and its size) before committing compute."""
    name = str(spec.get("name") or "sweep")
    combos = [
        {
            "label": combo_label(combo),
            "experiment_name": f"{name}/{combo_label(combo)}",
            "combo": combo,
        }
        for combo, _config in expand_sweep(spec)
    ]
    return {"name": name, "experiment_group": name, "count": len(combos), "combos": combos}


def expand_sweep(spec: Dict[str, Any]) -> List[Tuple[Dict[str, Any], EvaluationConfig]]:
    """Expand a sweep spec into ``[(combo, config), …]`` — one config per axis combination.

    Each config's ``experiment_group`` is set to the sweep name and its ``experiment_name``
    to ``<sweep>/<combo-label>``, so the run group is recoverable from the leaderboard.
    """
    _validate_spec(spec)
    name = str(spec.get("name") or "sweep")
    grid = GridSearch(_load_base_config(spec["base"]))
    for axis in spec["axes"]:
        grid.add_param(str(axis["path"]), list(axis["values"]))

    out: List[Tuple[Dict[str, Any], EvaluationConfig]] = []
    for combo, config in grid.generate_configs():
        config.experiment_group = name
        config.experiment_name = f"{name}/{combo_label(combo)}"
        out.append((combo, config))
    return out
