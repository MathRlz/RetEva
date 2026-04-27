"""Evaluation results dataclass and utilities.

This module provides structured result handling for evaluation runs,
including serialization, loading, and pretty printing capabilities.
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

from ..config import EvaluationConfig


@dataclass
class EvaluationResults:
    """Structured container for evaluation results.

    This class provides a typed interface for evaluation metrics, configuration,
    and metadata. It supports serialization to/from dictionaries and JSON files,
    as well as pretty printing for human-readable output.

    Attributes:
        metrics: Dictionary of evaluation metrics (e.g., MRR, WER, Recall@k).
        config: The EvaluationConfig used for this evaluation run.
        metadata: Additional metadata including timestamps, versions, etc.
            Common metadata fields:
            - start_time: ISO format timestamp of evaluation start
            - end_time: ISO format timestamp of evaluation end
            - duration_seconds: Total evaluation duration
            - evaluator_version: Version of the evaluator package
            - num_samples: Number of samples evaluated
            - pipeline_mode: Pipeline mode used (asr_text_retrieval, etc.)

    Example:
        Creating and saving results::

            >>> from evaluator import EvaluationConfig, EvaluationResults
            >>> config = EvaluationConfig.from_yaml("config.yaml")
            >>> metrics = {"MRR": 0.7523, "WER": 0.1234, "Recall@5": 0.8912}
            >>> metadata = {
            ...     "start_time": "2024-01-15T10:30:00",
            ...     "duration_seconds": 123.45,
            ...     "num_samples": 1000
            ... }
            >>> results = EvaluationResults(
            ...     metrics=metrics,
            ...     config=config,
            ...     metadata=metadata
            ... )
            >>> results.save("results/my_eval.json")

        Loading and inspecting results::

            >>> results = EvaluationResults.load("results/my_eval.json")
            >>> print(results)  # Pretty printed output
            >>> print(f"MRR: {results.metrics['MRR']:.4f}")
            >>> print(f"Duration: {results.metadata['duration_seconds']:.2f}s")

        Converting to dictionary for backward compatibility::

            >>> results_dict = results.to_dict()
            >>> # Dictionary contains all metrics at top level for compatibility
            >>> print(results_dict["MRR"])
            0.7523
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

        # Add creation timestamp if not present
        if "created_at" not in self.metadata:
            self.metadata["created_at"] = datetime.now().isoformat()

    def to_dict(self, include_config: bool = False) -> Dict[str, Any]:
        """Convert results to dictionary format.

        Returns a dictionary with metrics at top level and structured
        `_config`/`_metadata` sections.

        Args:
            include_config: If True, include full config in _config key.
                If False, include only minimal config info. Default: False.

        Returns:
            Dictionary with metrics and structured sections.

        Example:
            >>> results = EvaluationResults(
            ...     metrics={"MRR": 0.75, "WER": 0.12},
            ...     config=config,
            ...     metadata={"num_samples": 100}
            ... )
            >>> d = results.to_dict()
            >>> print(d["MRR"])
            0.75
            >>> print(d["_metadata"]["num_samples"])
            100

            >>> # With full config
            >>> d_full = results.to_dict(include_config=True)
            >>> print(d_full["_config"]["experiment_name"])
        """
        result = dict(self.metrics)

        # Add metadata section
        result["_metadata"] = dict(self.metadata)

        # Add config section
        if include_config:
            result["_config"] = self.config.to_dict(include_config=True)
        else:
            # Minimal config info
            result["_config"] = {
                "experiment_name": self.config.experiment_name,
                "pipeline_mode": self.config.model.pipeline_mode,
            }

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationResults":
        """Create EvaluationResults from dictionary.

        Accepts structured format with `_config` and `_metadata`.

        Args:
            data: Dictionary containing results data. Can be either:
                - {"_config": {...}, "_metadata": {...}, "MRR": 0.75, ...}

        Returns:
            EvaluationResults instance.

        Raises:
            ValueError: If data format is invalid or missing required fields.

        Example:
            >>> data = {
            ...     "MRR": 0.75,
            ...     "WER": 0.12,
            ...     "_config": {"experiment_name": "test"},
            ...     "_metadata": {"num_samples": 100}
            ... }
            >>> results = EvaluationResults.from_dict(data)
        """
        # Extract metadata and config
        metadata = data.get("_metadata", {})
        config_data = data.get("_config")

        # Extract metrics (everything except _config and _metadata)
        metrics = {
            k: v for k, v in data.items()
            if not k.startswith("_")
        }

        if config_data is None:
            raise ValueError("Missing _config section in result data")
        if isinstance(config_data, dict):
            config = EvaluationConfig.from_dict(config_data)
        else:
            raise ValueError(f"Invalid _config format: {type(config_data)}")

        return cls(
            metrics=metrics,
            config=config,
            metadata=metadata,
        )

    def save(self, path: Union[str, Path], indent: int = 2) -> None:
        """Save results to JSON file.

        Args:
            path: File path to save to. Parent directory will be created if needed.
            indent: JSON indentation level for pretty printing. Default: 2.

        Example:
            >>> results.save("results/my_eval.json")
            >>> results.save("results/compact.json", indent=None)  # Compact
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save with full config
        data = self.to_dict(include_config=True)

        with open(path, "w") as f:
            json.dump(data, f, indent=indent)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "EvaluationResults":
        """Load results from JSON file.

        Args:
            path: File path to load from.

        Returns:
            EvaluationResults instance.

        Raises:
            FileNotFoundError: If file doesn't exist.
            ValueError: If file format is invalid.

        Example:
            >>> results = EvaluationResults.load("results/my_eval.json")
            >>> print(results.metrics["MRR"])
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Results file not found: {path}")

        with open(path, "r") as f:
            data = json.load(f)

        return cls.from_dict(data)

    def __str__(self) -> str:
        """Pretty print results for human-readable output.

        Returns:
            Formatted string with experiment info, metrics, and metadata.

        Example:
            >>> print(results)
            Evaluation Results: my_experiment
            ================================================================================
            Pipeline Mode: asr_text_retrieval

            Metrics:
              MRR                 : 0.7523
              WER                 : 12.34%
              Recall@5            : 0.8912

            Metadata:
              num_samples         : 1000
              duration_seconds    : 123.45
        """
        lines = []

        # Header
        exp_name = self.config.experiment_name or "Unknown"
        lines.append(f"Evaluation Results: {exp_name}")
        lines.append("=" * 80)

        # Pipeline info
        pipeline_mode = self.config.model.pipeline_mode
        lines.append(f"Pipeline Mode: {pipeline_mode}")
        lines.append("")

        # Metrics section
        if self.metrics:
            lines.append("Metrics:")
            max_key_len = max(len(k) for k in self.metrics.keys())

            for key, value in sorted(self.metrics.items()):
                if isinstance(value, float):
                    # Format floats nicely
                    if "WER" in key or "CER" in key or "error" in key.lower():
                        # Error rates as percentages
                        formatted_value = f"{value:.2%}"
                    elif value < 1.0:
                        # Metrics like MRR, Recall, etc.
                        formatted_value = f"{value:.4f}"
                    else:
                        formatted_value = f"{value:.2f}"
                else:
                    formatted_value = str(value)

                lines.append(f"  {key:<{max_key_len}} : {formatted_value}")
            lines.append("")

        # Metadata section
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
        """Return detailed representation for debugging.

        Returns:
            String representation showing class name and key attributes.
        """
        return (
            f"EvaluationResults("
            f"experiment='{self.config.experiment_name}', "
            f"metrics={len(self.metrics)}, "
            f"metadata={len(self.metadata)})"
        )

    def get_metric(self, name: str, default: Any = None) -> Any:
        """Get a metric value by name with optional default.

        Args:
            name: Metric name (e.g., "MRR", "WER", "Recall@5").
            default: Default value if metric not found. Default: None.

        Returns:
            Metric value or default if not found.

        Example:
            >>> mrr = results.get_metric("MRR", 0.0)
            >>> recall = results.get_metric("Recall@10")
        """
        return self.metrics.get(name, default)

    def summary(self) -> str:
        """Return one-line summary of key metrics.

        Returns:
            Compact string with essential metrics.

        Example:
            >>> print(results.summary())
            MRR: 0.7523, WER: 12.34%, Recall@5: 0.8912
        """
        key_metrics = []

        # Common important metrics
        priority = ["MRR", "MAP", "WER", "CER", "Recall@5", "Recall@10", "NDCG@5"]

        for metric in priority:
            if metric in self.metrics:
                value = self.metrics[metric]
                if isinstance(value, float):
                    if "WER" in metric or "CER" in metric:
                        key_metrics.append(f"{metric}: {value:.2%}")
                    else:
                        key_metrics.append(f"{metric}: {value:.4f}")
                else:
                    key_metrics.append(f"{metric}: {value}")

        # Add any other metrics not in priority list
        for metric, value in self.metrics.items():
            if metric not in priority:
                if isinstance(value, float):
                    key_metrics.append(f"{metric}: {value:.4f}")
                else:
                    key_metrics.append(f"{metric}: {value}")

        return ", ".join(key_metrics) if key_metrics else "No metrics"
