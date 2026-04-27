"""Experiment tracking configuration."""
from dataclasses import dataclass
from typing import Optional


@dataclass
class TrackingConfig:
    """Configuration for experiment tracking.
    
    Attributes:
        enabled: Whether experiment tracking is enabled.
        backend: Tracking backend to use. Currently only "mlflow" is supported.
        mlflow_tracking_uri: MLflow tracking server URI. If None, uses local
            file-based tracking in the 'mlruns' directory.
        mlflow_experiment_name: Name for the MLflow experiment. If None, uses
            the evaluation experiment_name.
    """
    enabled: bool = False
    backend: str = "mlflow"
    mlflow_tracking_uri: Optional[str] = None
    mlflow_experiment_name: Optional[str] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        valid_backends = ["mlflow"]
        if self.backend not in valid_backends:
            raise ValueError(
                f"Invalid tracking backend: '{self.backend}'. "
                f"Valid options: {', '.join(valid_backends)}"
            )
