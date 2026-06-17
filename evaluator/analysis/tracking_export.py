"""Push an evaluation report to an experiment tracker (architecture-improvements §6).

The provenance block + the tidy metrics table map cleanly onto MLflow / Weights & Biases, so a
run plugs into existing lab infrastructure instead of replacing it. ``report_to_tracking_payload``
is the pure extraction (testable without either library); ``to_mlflow`` / ``to_wandb`` are thin,
import-guarded wrappers that log it.
"""

from __future__ import annotations

from typing import Any, Dict

from .report_export import _report_block, report_to_metrics_table


def report_to_tracking_payload(report: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Extract ``{metrics, params, tags}`` a tracker logs from a report (pure, no deps).

    - ``metrics``: ``{"<branch>/<metric>": mean}`` from the metrics table.
    - ``params``: identity from provenance (config hash, seed, resolved models, dataset
      fingerprint) — the things you filter/group runs by.
    - ``tags``: git commit, when present.
    """
    block = _report_block(report)
    prov = block.get("provenance") or {}

    metrics = {
        f"{r['branch']}/{r['metric']}": r["mean"]
        for r in report_to_metrics_table(report)
        if r["mean"] is not None
    }

    params: Dict[str, Any] = {}
    if prov.get("config_hash"):
        params["config_hash"] = prov["config_hash"]
    if prov.get("seed") is not None:
        params["seed"] = prov["seed"]
    for family, m in (prov.get("models") or {}).items():
        params[f"model.{family}"] = (
            m if isinstance(m, str) else (m.get("resolved") or m.get("type"))
        )
    for key, val in (prov.get("dataset") or {}).items():
        params[f"dataset.{key}"] = val

    tags = {"git_commit": prov["git_commit"]} if prov.get("git_commit") else {}
    return {"metrics": metrics, "params": params, "tags": tags}


def to_mlflow(report: Dict[str, Any], *, run_name=None, experiment=None) -> None:
    """Log the report's metrics + provenance to MLflow (needs ``mlflow``)."""
    try:
        import mlflow
    except ImportError as exc:  # pragma: no cover - optional dep
        raise ImportError("MLflow export needs mlflow (pip install mlflow)") from exc
    payload = report_to_tracking_payload(report)
    if experiment:
        mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(payload["params"])
        mlflow.log_metrics(payload["metrics"])
        for key, val in payload["tags"].items():
            mlflow.set_tag(key, val)


def to_wandb(report: Dict[str, Any], *, project=None, run_name=None) -> None:
    """Log the report's metrics + provenance to Weights & Biases (needs ``wandb``)."""
    try:
        import wandb
    except ImportError as exc:  # pragma: no cover - optional dep
        raise ImportError("W&B export needs wandb (pip install wandb)") from exc
    payload = report_to_tracking_payload(report)
    run = wandb.init(project=project, name=run_name, config=payload["params"])
    run.log(payload["metrics"])
    run.finish()
