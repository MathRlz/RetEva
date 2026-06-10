"""Terminal sink nodes: persist the aggregate ``report`` (architecture A6).

The ``aggregate`` node computes the ``report`` (per-branch scalars, deltas, provenance);
sink nodes *persist* it, keeping computation pure (the report is the single source of truth,
sinks are explicit graph I/O — mirrors ``dataset_sink``). ``leaderboard_sink`` writes the
report to a JSON sidecar (a leaderboard can index these); ``tracking_sink`` logs a summary
(MLflow/other tracking is optional). Both no-op without a ``report``.

These are graph-native persistence; the runner's existing out-of-graph leaderboard ingest
(``services/evaluation_service``) stays the default, so the sinks are opt-in (added to a graph
explicitly) and never double-write.

See ``evaluator-architecture.md`` §8.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from ..logging_config import get_logger

logger = get_logger(__name__)


def write_report_json(report: Dict[str, Any], output_dir: str, name: str) -> str:
    """Write ``report`` to ``<output_dir>/report_<name>.json``; return the path."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"report_{name}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, default=str, indent=2)
    return path


def summarize_report(report: Dict[str, Any]) -> Dict[str, float]:
    """Flatten a report's per-branch metric means to ``{"<branch>/<metric>": mean}``."""
    flat: Dict[str, float] = {}
    for branch, metrics in (report.get("branches") or {}).items():
        for metric, stats in metrics.items():
            if isinstance(stats, dict) and "mean" in stats:
                flat[f"{branch}/{metric}"] = stats["mean"]
    return flat


def log_report_to_tracking(
    report: Dict[str, Any], tracker: Optional[Any] = None
) -> Dict[str, float]:
    """Log the report's flattened metrics to ``tracker`` (if given) and the run log.

    Returns the flattened metrics (also the unit-test surface). A real MLflow tracker is
    optional — duck-typed ``tracker.log_metrics(dict)``."""
    flat = summarize_report(report)
    if tracker is not None and hasattr(tracker, "log_metrics"):
        try:
            tracker.log_metrics(flat)
        except Exception as exc:  # never let tracking break a run
            logger.warning("tracking_sink: log_metrics failed: %s", exc)
    logger.info("tracking_sink: %d report metrics — %s", len(flat), flat)
    return flat
