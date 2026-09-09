"""Shared MLflow call helpers. ``mlflow`` is an optional dependency imported lazily by each
caller, so every function here takes the already-imported module as its first argument
rather than importing it itself.
"""

from typing import Any, Dict, Iterable, Optional


def set_experiment(mlflow, experiment: Optional[str]) -> None:
    """Set the active MLflow experiment; no-op when ``experiment`` is falsy."""
    if experiment:
        mlflow.set_experiment(experiment)


def start_run(mlflow, run_name: Optional[str] = None):
    """Start an MLflow run; returns the run context (usable as a ``with`` target)."""
    return mlflow.start_run(run_name=run_name)


def log_params(mlflow, params: Dict[str, Any]) -> None:
    """Log a flat params dict; no-op when empty."""
    if params:
        mlflow.log_params(params)


def log_metrics(mlflow, metrics: Dict[str, float], step: Optional[int] = None) -> None:
    """Log a flat metrics dict; no-op when empty. ``step`` marks time-series metrics."""
    if not metrics:
        return
    if step is not None:
        mlflow.log_metrics(metrics, step=step)
    else:
        mlflow.log_metrics(metrics)


def log_artifacts(mlflow, artifacts: Optional[Iterable[str]]) -> None:
    """Log each artifact path; no-op when ``artifacts`` is falsy."""
    for path in artifacts or ():
        mlflow.log_artifact(path)


def log_tags(mlflow, tags: Optional[Dict[str, str]]) -> None:
    """Set each tag; no-op when ``tags`` is falsy."""
    for key, val in (tags or {}).items():
        mlflow.set_tag(key, val)


def end_run(mlflow) -> None:
    mlflow.end_run()


def log_run(
    mlflow,
    *,
    experiment: Optional[str] = None,
    run_name: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
    metrics: Optional[Dict[str, float]] = None,
    artifacts: Optional[Iterable[str]] = None,
    tags: Optional[Dict[str, str]] = None,
) -> None:
    """One-shot MLflow log: for a single report/summary, not the live tracker's incremental
    path (that drives these steps individually)."""
    set_experiment(mlflow, experiment)
    with start_run(mlflow, run_name=run_name):
        log_params(mlflow, params or {})
        log_metrics(mlflow, metrics or {})
        log_artifacts(mlflow, artifacts)
        log_tags(mlflow, tags)
