"""Evaluation results dataclass and utilities.

This module provides structured result handling for evaluation runs,
including serialization, loading, and pretty printing capabilities.

Result-type boundary (T1). Two result types coexist on purpose, at different layers:

- :class:`RunResults` (``evaluation/result_schema.py``) is the **internal** result: a
  ``dict`` subclass of ~40 metric keys with typed accessors, produced by the executor
  (``run_graph`` / ``run_from_bundle``) and consumed inside the evaluation engine.
- :class:`EvaluationResults` (here) is the **public** result: a dataclass that wraps the
  metrics dict together with the ``EvaluationConfig`` and run metadata (timestamps,
  versions, sample counts) and adds JSON file save/load + pretty printing. The public API
  (``api.py`` / ``public_api.py``) builds an ``EvaluationResults`` from a ``RunResults``.

So: ``RunResults`` = the metrics payload; ``EvaluationResults`` = that payload + provenance
+ I/O, for callers. They are not merged because the internal dict must stay a plain mapping
(its own serialized form at the engine edge), while the public type carries config/metadata.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Union

from ..config import EvaluationConfig

# Canonical headline-metric priority — the report keys shown first / most prominently
# (the textual ``summary()`` here, the web status chips, any "primary metric" UI). One list so
# the console summary and the UI never disagree about which metrics lead. Keys absent from a
# given run are simply skipped.
HEADLINE_METRICS = (
    "MRR", "MAP", "Recall@1", "Recall@5", "Recall@10", "NDCG@5", "WER", "CER",
)


@dataclass
class EvaluationResults:
    """Structured container for evaluation results.

    Attributes:
        metrics: Dictionary of evaluation metrics (e.g., MRR, WER, Recall@k).
        config: The EvaluationConfig used for this evaluation run.
        metadata: Additional run metadata (start_time/end_time, duration_seconds,
            evaluator_version, num_samples, pipeline_mode, ...).
    """

    metrics: Dict[str, Any]
    config: EvaluationConfig
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate and normalize fields after initialization."""
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
        """Return a dict with metrics at top level plus structured
        ``_config``/``_metadata`` sections.

        Args:
            include_config: If True, include the full config under ``_config``;
                otherwise only experiment_name/pipeline_mode. Default: False.
        """
        result = dict(self.metrics)

        result["_metadata"] = dict(self.metadata)

        if include_config:
            result["_config"] = self.config.to_dict(include_config=True)
        else:
            result["_config"] = {
                "experiment_name": self.config.experiment_name,
                "pipeline_mode": self.config.graph_template,
            }

        return result

    def save(self, path: Union[str, Path], indent: int = 2) -> None:
        """Save results (with full config) to a JSON file.

        Args:
            path: File path to save to. Parent directory is created if needed.
            indent: JSON indentation level. Default: 2.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = self.to_dict(include_config=True)

        with open(path, "w") as f:
            json.dump(data, f, indent=indent)

    def __str__(self) -> str:
        """Pretty print results: experiment info, metrics, and metadata."""
        lines = []

        exp_name = self.config.experiment_name or "Unknown"
        lines.append(f"Evaluation Results: {exp_name}")
        lines.append("=" * 80)

        pipeline_mode = self.config.graph_template
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
        """Return detailed representation for debugging."""
        return (
            f"EvaluationResults("
            f"experiment='{self.config.experiment_name}', "
            f"metrics={len(self.metrics)}, "
            f"metadata={len(self.metadata)})"
        )

    def get_metric(self, name: str, default: Any = None) -> Any:
        """Return the metric value for ``name`` (e.g. "MRR", "Recall@5"),
        or ``default`` if absent."""
        return self.metrics.get(name, default)

    def to_dataframe(self):
        """Tidy per-branch metric rows as a pandas ``DataFrame`` (C2): one row per
        ``(branch, metric)`` with mean / ci_lower / ci_upper / n from the keyed ``report``
        — the notebook-side sibling of ``evaluator export -f metrics-table``. Empty
        DataFrame when the run carries no keyed report.

        Example:
            >>> df = results.to_dataframe()
            >>> df[df.metric == "recall@5"].set_index("branch")["mean"]
        """
        import pandas as pd

        from ..analysis.report_export import report_to_metrics_table

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
