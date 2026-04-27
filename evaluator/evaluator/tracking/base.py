"""Base tracker protocol and no-op implementation.

This module defines the tracking interface that all tracker backends must implement,
plus a no-op tracker for when tracking is disabled.
"""

from typing import Dict, Any, Optional, Protocol, runtime_checkable


@runtime_checkable
class BaseTracker(Protocol):
    """Protocol defining the tracker interface.
    
    All tracker backends must implement these methods for consistent
    experiment tracking across different backends (MLflow, W&B, etc.).
    """
    
    def start_run(self, run_name: Optional[str] = None) -> None:
        """Start a new tracking run.
        
        Args:
            run_name: Optional name for this run.
        """
        ...
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log configuration parameters.
        
        Args:
            params: Dictionary of parameter names to values.
        """
        ...
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log evaluation metrics.
        
        Args:
            metrics: Dictionary of metric names to values.
            step: Optional step number for time-series metrics.
        """
        ...
    
    def log_artifact(self, path: str) -> None:
        """Log an artifact file.
        
        Args:
            path: Path to the artifact file to log.
        """
        ...
    
    def end_run(self) -> None:
        """End the current tracking run."""
        ...
    
    def __enter__(self) -> "BaseTracker":
        """Context manager entry."""
        ...
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        ...


class NoOpTracker:
    """No-op tracker for when tracking is disabled.
    
    This tracker implements the BaseTracker protocol but does nothing,
    allowing code to use the tracking interface without actually
    logging anything.
    
    Examples:
        Using as a context manager::
        
            >>> tracker = NoOpTracker()
            >>> with tracker:
            ...     tracker.log_params({"batch_size": 32})
            ...     tracker.log_metrics({"MRR": 0.75})
            >>> # Nothing happens, no data is logged
    """
    
    def __init__(self, experiment_name: Optional[str] = None, **kwargs: Any) -> None:
        """Initialize no-op tracker.
        
        Args:
            experiment_name: Ignored.
            **kwargs: Ignored.
        """
        pass
    
    def start_run(self, run_name: Optional[str] = None) -> None:
        """No-op: does nothing."""
        pass
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """No-op: does nothing."""
        pass
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """No-op: does nothing."""
        pass
    
    def log_artifact(self, path: str) -> None:
        """No-op: does nothing."""
        pass
    
    def end_run(self) -> None:
        """No-op: does nothing."""
        pass
    
    def __enter__(self) -> "NoOpTracker":
        """Context manager entry - does nothing."""
        return self
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit - does nothing."""
        pass
