"""Evaluation results: :class:`EvaluationResults` (public) wraps the metrics dict with the
run's :class:`EvaluationConfig` + metadata, adding JSON save/load + pretty printing.
:class:`~evaluator.evaluation.result_schema.RunResults` stays a plain mapping (the engine's
serialized form) and is not merged into this class for that reason."""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Union

from ..config import EvaluationConfig
from ..analysis.report_export import report_to_metrics_table

# Headline-metric display order — shared by ``summary()`` and the UI so they never disagree
# about which metrics lead. Keys absent from a given run are simply skipped.
HEADLINE_METRICS = (
    "MRR", "MAP", "Recall@1", "Recall@5", "Recall@10", "NDCG@5", "WER", "CER",
)


@dataclass
class EvaluationResults:
    """``metadata`` holds run metadata: start_time/end_time, duration_seconds,
    evaluator_version, num_samples, pipeline_mode, ...
    """

    metrics: Dict[str, Any]
    config: EvaluationConfig
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.metrics, dict):
            raise TypeError(f"metrics must be a dict, got {type(self.metrics)}")

        if not isinstance(self.config, EvaluationConfig):
            raise TypeError(
                f"config must be an EvaluationConfig, got {type(self.config)}"
            )

        if not isinstance(self.metadata, dict):
            raise TypeError(f"metadata must be a dict, got {type(self.metadata)}")

        if "created_at" not in self.metadata:
            self.metadata["created_at"] = datetime.now().isoformat()

    def to_dict(self, include_config: bool = False) -> Dict[str, Any]:
        """Dict with metrics at top level plus ``_config``/``_metadata`` sections.
        ``include_config=True`` includes the full config under ``_config``; otherwise just
        experiment_name/pipeline_mode."""
        result = dict(self.metrics)

        result["_metadata"] = dict(self.metadata)

        if include_config:
            result["_config"] = self.config.to_dict(include_config=True)
        else:
            result["_config"] = {
                "experiment_name": self.config.experiment_name,
                "pipeline_mode": self.config.graph_label,
            }

        return result

    def save(self, path: Union[str, Path], indent: int = 2) -> None:
        """Save results (with full config) to a JSON file; the parent directory is
        created if needed."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = self.to_dict(include_config=True)

        with open(path, "w") as f:
            json.dump(data, f, indent=indent)

    def __str__(self) -> str:
        lines = []

        exp_name = self.config.experiment_name or "Unknown"
        lines.append(f"Evaluation Results: {exp_name}")
        lines.append("=" * 80)

        pipeline_mode = self.config.graph_label
        lines.append(f"Pipeline Mode: {pipeline_mode}")
        lines.append("")

        if self.metrics:
            lines.append("Metrics:")
            max_key_len = max(len(k) for k in self.metrics.keys())

            for key, value in sorted(self.metrics.items()):
                if isinstance(value, float):
                    if "WER" in key or "CER" in key or "error" in key.lower():
                        formatted_value = f"{value:.2%}"
                    elif value < 1.0:
                        formatted_value = f"{value:.4f}"
                    else:
                        formatted_value = f"{value:.2f}"
                else:
                    formatted_value = str(value)

                lines.append(f"  {key:<{max_key_len}} : {formatted_value}")
            lines.append("")

        if self.metadata:
            lines.append("Metadata:")
            max_key_len = max(len(k) for k in self.metadata.keys())

            for key, value in sorted(self.metadata.items()):
                if isinstance(value, float):
                    formatted_value = f"{value:.2f}"
                else:
                    formatted_value = str(value)

                lines.append(f"  {key:<{max_key_len}} : {formatted_value}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"EvaluationResults("
            f"experiment='{self.config.experiment_name}', "
            f"metrics={len(self.metrics)}, "
            f"metadata={len(self.metadata)})"
        )

    def get_metric(self, name: str, default: Any = None) -> Any:
        return self.metrics.get(name, default)

    def to_dataframe(self):
        """Tidy per-branch metric rows as a pandas ``DataFrame``: one row per
        ``(branch, metric)`` with mean / ci_lower / ci_upper / n from the keyed ``report``.
        Empty DataFrame when the run carries no keyed report."""
        import pandas as pd

        return pd.DataFrame(report_to_metrics_table(self.metrics))

    def summary(self) -> str:
        """Return a one-line summary: headline metrics first, then the rest."""
        key_metrics = []

        for metric in HEADLINE_METRICS:
            if metric in self.metrics:
                value = self.metrics[metric]
                if isinstance(value, float):
                    if "WER" in metric or "CER" in metric:
                        key_metrics.append(f"{metric}: {value:.2%}")
                    else:
                        key_metrics.append(f"{metric}: {value:.4f}")
                else:
                    key_metrics.append(f"{metric}: {value}")

        for metric, value in self.metrics.items():
            if metric not in HEADLINE_METRICS:
                if isinstance(value, float):
                    key_metrics.append(f"{metric}: {value:.4f}")
                else:
                    key_metrics.append(f"{metric}: {value}")

        return ", ".join(key_metrics) if key_metrics else "No metrics"
