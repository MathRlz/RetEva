"""MLflow experiment tracking implementation.

This module provides MLflow integration for tracking evaluation experiments,
including parameters, metrics, and artifacts.
"""

from typing import Dict, Any, Optional
from pathlib import Path

from ..logging_config import get_logger

logger = get_logger(__name__)


class MLflowTracker:
    """MLflow experiment tracker for evaluation runs.
    
    Wraps MLflow functionality for logging parameters, metrics, and artifacts
    during evaluation. Supports both local file-based tracking and remote
    tracking servers.
    
    Attributes:
        experiment_name: Name of the MLflow experiment.
        tracking_uri: MLflow tracking server URI or local path.
        _run_active: Whether a run is currently active.
    
    Examples:
        Basic usage with local tracking::
        
            >>> tracker = MLflowTracker("my_experiment")
            >>> with tracker:
            ...     tracker.log_params({"batch_size": 32, "model": "whisper"})
            ...     tracker.log_metrics({"MRR": 0.75, "MAP": 0.68})
        
        With a tracking server::
        
            >>> tracker = MLflowTracker(
            ...     experiment_name="production_eval",
            ...     tracking_uri="http://mlflow-server:5000"
            ... )
            >>> tracker.start_run(run_name="eval_run_001")
            >>> tracker.log_params({"k": 10})
            >>> tracker.log_metrics({"Recall@5": 0.82})
            >>> tracker.end_run()
        
        Logging artifacts::
        
            >>> with MLflowTracker("artifact_demo") as tracker:
            ...     tracker.log_artifact("results/metrics.json")
            ...     tracker.log_artifact("results/confusion_matrix.png")
    """
    
    def __init__(
        self,
        experiment_name: str,
        tracking_uri: Optional[str] = None
    ) -> None:
        """Initialize MLflow tracker.
        
        Args:
            experiment_name: Name for the MLflow experiment. Will be created
                if it doesn't exist.
            tracking_uri: MLflow tracking server URI. If None, uses local
                file-based tracking in the 'mlruns' directory.
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self._run_active = False
        self._mlflow = None
    
    def _ensure_mlflow(self) -> Any:
        """Lazily import and configure MLflow.
        
        Returns:
            The mlflow module.
            
        Raises:
            ImportError: If mlflow is not installed.
        """
        if self._mlflow is None:
            try:
                import mlflow
                self._mlflow = mlflow
                
                if self.tracking_uri:
                    mlflow.set_tracking_uri(self.tracking_uri)
                    logger.debug(f"MLflow tracking URI set to: {self.tracking_uri}")
                
                mlflow.set_experiment(self.experiment_name)
                logger.debug(f"MLflow experiment set to: {self.experiment_name}")
                
            except ImportError as e:
                raise ImportError(
                    "MLflow is required for experiment tracking. "
                    "Install it with: pip install mlflow"
                ) from e
        
        return self._mlflow
    
    def start_run(self, run_name: Optional[str] = None) -> None:
        """Start a new MLflow run.
        
        Args:
            run_name: Optional name for the run. If None, MLflow generates one.
        """
        mlflow = self._ensure_mlflow()
        
        if self._run_active:
            logger.warning("A run is already active. Ending it before starting a new one.")
            self.end_run()
        
        mlflow.start_run(run_name=run_name)
        self._run_active = True
        logger.info(f"Started MLflow run: {run_name or '(auto-generated)'}")
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters to MLflow.
        
        Args:
            params: Dictionary of parameter names to values. Values are
                converted to strings for MLflow compatibility.
        """
        if not self._run_active:
            logger.warning("No active run. Call start_run() first.")
            return
        
        mlflow = self._ensure_mlflow()
        
        # Flatten nested dicts and convert values to strings
        flat_params = self._flatten_params(params)
        
        # MLflow has a limit of 500 params per batch
        param_items = list(flat_params.items())
        batch_size = 100
        
        for i in range(0, len(param_items), batch_size):
            batch = dict(param_items[i:i + batch_size])
            mlflow.log_params(batch)
        
        logger.debug(f"Logged {len(flat_params)} parameters to MLflow")
    
    def _flatten_params(
        self, 
        params: Dict[str, Any], 
        parent_key: str = "", 
        sep: str = "."
    ) -> Dict[str, str]:
        """Flatten nested parameter dictionaries.
        
        Args:
            params: Nested parameter dictionary.
            parent_key: Prefix for flattened keys.
            sep: Separator between nested keys.
            
        Returns:
            Flattened dictionary with string values.
        """
        items: Dict[str, str] = {}
        
        for key, value in params.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else key
            
            if isinstance(value, dict):
                items.update(self._flatten_params(value, new_key, sep))
            else:
                # Convert to string, truncating if too long (MLflow limit is 500 chars)
                str_value = str(value)
                if len(str_value) > 500:
                    str_value = str_value[:497] + "..."
                items[new_key] = str_value
        
        return items
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics to MLflow.
        
        Args:
            metrics: Dictionary of metric names to float values.
            step: Optional step number for time-series metrics.
        """
        if not self._run_active:
            logger.warning("No active run. Call start_run() first.")
            return
        
        mlflow = self._ensure_mlflow()
        
        # Filter to only numeric metrics
        numeric_metrics = {
            k: float(v) for k, v in metrics.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
        
        if step is not None:
            mlflow.log_metrics(numeric_metrics, step=step)
        else:
            mlflow.log_metrics(numeric_metrics)
        
        logger.debug(f"Logged {len(numeric_metrics)} metrics to MLflow")
    
    def log_artifact(self, path: str) -> None:
        """Log an artifact file to MLflow.
        
        Args:
            path: Path to the artifact file. Must exist.
        """
        if not self._run_active:
            logger.warning("No active run. Call start_run() first.")
            return
        
        artifact_path = Path(path)
        if not artifact_path.exists():
            logger.warning(f"Artifact path does not exist: {path}")
            return
        
        mlflow = self._ensure_mlflow()
        mlflow.log_artifact(str(artifact_path))
        logger.debug(f"Logged artifact: {path}")
    
    def end_run(self) -> None:
        """End the current MLflow run."""
        if not self._run_active:
            return
        
        mlflow = self._ensure_mlflow()
        mlflow.end_run()
        self._run_active = False
        logger.info("Ended MLflow run")
    
    def __enter__(self) -> "MLflowTracker":
        """Context manager entry - starts a run."""
        self.start_run()
        return self
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit - ends the run."""
        self.end_run()
