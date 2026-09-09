"""Experiment tracking configuration."""
from dataclasses import dataclass
from typing import Optional


@dataclass
class TrackingConfig:
    """Configuration for experiment tracking."""
    enabled: bool = False
    backend: str = "mlflow"  # only "mlflow" is currently supported
    mlflow_tracking_uri: Optional[str] = None  # None = local file-based tracking ('mlruns' dir)
    mlflow_experiment_name: Optional[str] = None  # None = uses the evaluation experiment_name

    def __post_init__(self):
        valid_backends = ["mlflow"]
        if self.backend not in valid_backends:
            raise ValueError(
                f"Invalid tracking backend: '{self.backend}'. "
                f"Valid options: {', '.join(valid_backends)}"
            )
