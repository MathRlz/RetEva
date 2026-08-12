"""No-op tracker for when tracking is disabled.

Tracker backends (MLflowTracker, this) share a duck-typed interface:
start_run / log_params / log_metrics / log_artifact / end_run + context manager.
"""

from typing import Any, Dict, Optional


class NoOpTracker:
    """Implements the tracker interface but does nothing, so code can use
    tracking unconditionally."""

    def __init__(self, experiment_name: Optional[str] = None, **kwargs: Any) -> None:
        pass

    def start_run(self, run_name: Optional[str] = None) -> None:
        pass

    def log_params(self, params: Dict[str, Any]) -> None:
        pass

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        pass

    def log_artifact(self, path: str) -> None:
        pass

    def end_run(self) -> None:
        pass

    def __enter__(self) -> "NoOpTracker":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass
