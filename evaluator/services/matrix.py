"""Run-matrix / comparison helpers (extracted from evaluation_service)."""

from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence

from ..config import EvaluationConfig
from ..errors import ConfigurationError
from . import evaluation_service


def _apply_setup_overrides(
    config: EvaluationConfig, overrides: Dict[str, Any]
) -> EvaluationConfig:
    """Apply setup overrides to a config copy."""
    nested_sections = {
        "cache",
        "logging",
        "model",
        "data",
        "audio_synthesis",
        "augmentation",
        "llm_server",
        "judge",
        "query_optimization",
        "embedding_fusion",
        "vector_db",
        "tracking",
        "device_pool",
        "service_runtime",
        "answer_generation",
        "llm",
    }
    section_aliases = {"vector": "vector_db"}
    top_level_fields = {
        "experiment_name",
        "output_dir",
        "checkpoint_enabled",
        "checkpoint_interval",
        "resume_from_checkpoint",
        "parallel_enabled",
        "num_parallel_workers",
    }

    for key, value in overrides.items():
        # Dotted-path format: "model.asr_device" → config.model.asr_device
        if "." in key:
            section_name, _, sub_key = key.partition(".")
            if section_name in nested_sections or section_name in section_aliases:
                target = section_aliases.get(section_name, section_name)
                section = getattr(config, target)
                setattr(section, sub_key, value)
                continue
            if section_name in top_level_fields or not sub_key:
                raise ConfigurationError(f"Unsupported dotted override key: {key}")
            raise ConfigurationError(
                f"Unknown section in dotted override: {section_name!r}"
            )

        if key in top_level_fields:
            setattr(config, key, value)
            continue
        if key in nested_sections and isinstance(value, dict):
            section = getattr(config, key)
            for sub_key, sub_value in value.items():
                setattr(section, sub_key, sub_value)
            continue
        # Legacy underscore-prefix format (kept for backward compatibility)
        matched_section = None
        sub_key = None
        for candidate in sorted(nested_sections, key=len, reverse=True):
            prefix = f"{candidate}_"
            if key.startswith(prefix):
                matched_section = candidate
                sub_key = key[len(prefix) :]
                break
        if matched_section is None:
            for alias, target in section_aliases.items():
                prefix = f"{alias}_"
                if key.startswith(prefix):
                    matched_section = target
                    sub_key = key[len(prefix) :]
                    break
        if matched_section is not None and sub_key:
            section = getattr(config, matched_section)
            setattr(section, sub_key, value)
            continue
        raise ConfigurationError(f"Unsupported setup override key: {key}")

    return config


def run_evaluation_matrix(
    base_config: EvaluationConfig,
    test_setups: Sequence[Dict[str, Any]],
    baseline_setup_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Run multiple test setups on the same dataset and return aggregated outputs.

    Args:
        base_config: Base configuration shared across all setups.
        test_setups: List of setup dicts with optional ``setup_id`` and ``overrides``.
        baseline_setup_id: ``setup_id`` of the setup to use as the comparison
            baseline.  Defaults to the first setup in ``test_setups``.
    """
    runs: List[Dict[str, Any]] = []

    for idx, setup in enumerate(test_setups):
        setup_id = setup.get("setup_id") or setup.get("name") or f"setup_{idx + 1:03d}"
        overrides = setup.get("overrides", {})
        if not isinstance(overrides, dict):
            raise ConfigurationError(
                f"Setup '{setup_id}' must provide dict overrides, got {type(overrides).__name__}"
            )

        config_variant = deepcopy(base_config)
        config_variant = _apply_setup_overrides(config_variant, overrides)
        if "experiment_name" not in overrides:
            config_variant.experiment_name = (
                f"{base_config.experiment_name}__{setup_id}"
            )

        result = evaluation_service.run_evaluation(config_variant)
        runs.append(
            {
                "setup_id": setup_id,
                "result": result.to_dict(include_config=True),
            }
        )

    comparison = _build_comparison_bundle(runs, baseline_setup_id=baseline_setup_id)

    return {
        "base_experiment_name": base_config.experiment_name,
        "num_setups": len(test_setups),
        "runs": runs,
        "comparison": comparison,
    }


def _build_comparison_bundle(
    runs: Sequence[Dict[str, Any]],
    baseline_setup_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a stable comparison artifact (baseline deltas + leaderboard)."""
    if not runs:
        return {"baseline_setup_id": None, "metric_deltas": [], "leaderboard": []}

    baseline_run = (
        next(
            (r for r in runs if str(r["setup_id"]) == baseline_setup_id),
            runs[0],
        )
        if baseline_setup_id
        else runs[0]
    )
    baseline_setup_id = str(baseline_run["setup_id"])
    baseline_metrics = baseline_run["result"]

    def numeric_metrics(result_dict: Dict[str, Any]) -> Dict[str, float]:
        values: Dict[str, float] = {}
        for k, v in result_dict.items():
            if k.startswith("_") or isinstance(v, bool):
                continue
            if isinstance(v, (int, float)):
                values[k] = float(v)
        return values

    baseline_numeric = numeric_metrics(baseline_metrics)
    metric_deltas: List[Dict[str, Any]] = []
    leaderboard_rows: List[Dict[str, Any]] = []

    for row in runs:
        setup_id = str(row["setup_id"])
        current_numeric = numeric_metrics(row["result"])
        shared = sorted(set(baseline_numeric).intersection(current_numeric))
        deltas = {
            metric: current_numeric[metric] - baseline_numeric[metric]
            for metric in shared
        }
        metric_deltas.append(
            {
                "setup_id": setup_id,
                "deltas_vs_baseline": deltas,
            }
        )
        leaderboard_rows.append(
            {
                "setup_id": setup_id,
                "metrics": current_numeric,
            }
        )

    ranking_metric = "MRR"
    if not any(ranking_metric in row["metrics"] for row in leaderboard_rows):
        metric_names: set[str] = set()
        for row in leaderboard_rows:
            metric_names.update(row["metrics"].keys())
        ranking_metric = sorted(metric_names)[0] if metric_names else "MRR"

    leaderboard = sorted(
        leaderboard_rows,
        key=lambda r: r["metrics"].get(ranking_metric, float("-inf")),
        reverse=True,
    )

    return {
        "baseline_setup_id": baseline_setup_id,
        "ranking_metric": ranking_metric,
        "metric_deltas": metric_deltas,
        "leaderboard": leaderboard,
    }
