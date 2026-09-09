"""MLflow experiment tracking implementation: parameters, metrics, and artifacts for
evaluation runs.
"""

from typing import Dict, Any, Optional
from pathlib import Path

from ..logging_config import get_logger
from . import mlflow_calls

logger = get_logger(__name__)


class MLflowTracker:
    """MLflow experiment tracker for evaluation runs; supports both local file-based
    tracking and remote tracking servers."""

    def __init__(
        self,
        experiment_name: str,
        tracking_uri: Optional[str] = None
    ) -> None:
        """The experiment is created if it doesn't exist; ``tracking_uri`` None
        means local file-based tracking in the 'mlruns' directory."""
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self._run_active = False
        self._mlflow = None

    def _ensure_mlflow(self) -> Any:
        """Lazily import and configure MLflow; returns the mlflow module."""
        if self._mlflow is None:
            try:
                import mlflow
                self._mlflow = mlflow

                if self.tracking_uri:
                    mlflow.set_tracking_uri(self.tracking_uri)
                    logger.debug(f"MLflow tracking URI set to: {self.tracking_uri}")

                mlflow_calls.set_experiment(mlflow, self.experiment_name)
                logger.debug(f"MLflow experiment set to: {self.experiment_name}")

            except ImportError as e:
                raise ImportError(
                    "MLflow is required for experiment tracking. "
                    "Install it with: pip install mlflow"
                ) from e

        return self._mlflow

    def start_run(self, run_name: Optional[str] = None) -> None:
        """Start a new MLflow run; MLflow generates a name when ``run_name`` is None."""
        mlflow = self._ensure_mlflow()

        if self._run_active:
            logger.warning("A run is already active. Ending it before starting a new one.")
            self.end_run()

        mlflow_calls.start_run(mlflow, run_name=run_name)
        self._run_active = True
        logger.info(f"Started MLflow run: {run_name or '(auto-generated)'}")

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters to MLflow; nested dicts are flattened and values are
        converted to strings for MLflow compatibility."""
        if not self._run_active:
            logger.warning("No active run. Call start_run() first.")
            return

        mlflow = self._ensure_mlflow()

        flat_params = self._flatten_params(params)

        # MLflow has a limit of 500 params per batch
        param_items = list(flat_params.items())
        batch_size = 100

        for i in range(0, len(param_items), batch_size):
            batch = dict(param_items[i:i + batch_size])
            mlflow_calls.log_params(mlflow, batch)

        logger.debug(f"Logged {len(flat_params)} parameters to MLflow")

    def _flatten_params(
        self,
        params: Dict[str, Any],
        parent_key: str = "",
        sep: str = "."
    ) -> Dict[str, str]:
        """Flatten nested parameter dicts into ``sep``-joined keys with string values."""
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
        """Log metrics to MLflow; non-numeric values are dropped. ``step`` is an
        optional step number for time-series metrics."""
        if not self._run_active:
            logger.warning("No active run. Call start_run() first.")
            return

        mlflow = self._ensure_mlflow()

        numeric_metrics = {
            k: float(v) for k, v in metrics.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }

        mlflow_calls.log_metrics(mlflow, numeric_metrics, step=step)

        logger.debug(f"Logged {len(numeric_metrics)} metrics to MLflow")

    def log_artifact(self, path: str) -> None:
        """Log an artifact file to MLflow; a missing path is skipped with a warning."""
        if not self._run_active:
            logger.warning("No active run. Call start_run() first.")
            return

        artifact_path = Path(path)
        if not artifact_path.exists():
            logger.warning(f"Artifact path does not exist: {path}")
            return

        mlflow = self._ensure_mlflow()
        mlflow_calls.log_artifacts(mlflow, [str(artifact_path)])
        logger.debug(f"Logged artifact: {path}")

    def end_run(self) -> None:
        if not self._run_active:
            return

        mlflow = self._ensure_mlflow()
        mlflow_calls.end_run(mlflow)
        self._run_active = False
        logger.info("Ended MLflow run")

    def __enter__(self) -> "MLflowTracker":
        self.start_run()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.end_run()
