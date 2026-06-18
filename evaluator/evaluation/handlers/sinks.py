"""Sink stage handlers: dataset / leaderboard / tracking output nodes.

Moved verbatim from the former ``evaluation/phased.py`` (Phase 1, X5). Each handler
registers itself via ``@register_stage_handler`` at import time.
"""

from __future__ import annotations

from ..stage_registry import register_stage_handler
from ...logging_config import get_logger
from ..executor.state import RunState

logger = get_logger(__name__)


@register_stage_handler("sink", self_timed=True)
def _stage_sink(s: RunState) -> None:
    """The ``sink`` operator: dispatch by target to finalize / aggregate / dataset /
    leaderboard / tracking. Bodies unchanged (finalize lives in the rag handlers,
    aggregate in the metrics handlers)."""
    from .rag import _stage_finalize
    from .metrics import _stage_aggregate
    from ._dispatch import dispatch_operator

    return dispatch_operator("sink", {
        "finalize": _stage_finalize,
        "aggregate": _stage_aggregate,
        "dataset_sink": _stage_dataset_sink,
        "leaderboard_sink": _stage_leaderboard_sink,
        "tracking_sink": _stage_tracking_sink,
    }, s)


def _stage_dataset_sink(s: RunState) -> None:
    """Persist the run's prepared/generated data (questions + synthesized audio_path +
    generated answers) to JSONL — enables benchmark-prep + synthetic-data graphs.
    No-op when config/sink is absent or disabled."""
    cfg = getattr(s.config, "dataset_sink", None) if s.config is not None else None
    if cfg is None or not getattr(cfg, "enabled", False):
        return
    questions = getattr(s.dataset, "questions", None)
    if not questions:
        return
    import json
    import os

    path = cfg.path
    if not path:
        out_dir = getattr(s.config, "output_dir", ".")
        os.makedirs(out_dir, exist_ok=True)
        name = getattr(s.config, "experiment_name", "run")
        path = os.path.join(out_dir, f"prepared_{name}.jsonl")
    else:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    gen_by_qid: dict = {}
    if cfg.include_generated:
        for trace in s.results.get("query_traces") or []:
            if trace.get("generated_answer"):
                gen_by_qid[str(trace.get("query_id"))] = trace["generated_answer"]

    written = 0
    with open(path, "w", encoding="utf-8") as fh:
        for idx, q in enumerate(questions):
            qid = str(getattr(q, "question_id", idx))
            rec = {
                "question_id": qid,
                "question_text": getattr(q, "question_text", None),
                "groundtruth_doc_ids": getattr(q, "groundtruth_doc_ids", None),
            }
            if cfg.include_audio and getattr(q, "audio_path", None):
                rec["audio_path"] = q.audio_path
            if cfg.include_generated and qid in gen_by_qid:
                rec["generated_answer"] = gen_by_qid[qid]
            fh.write(json.dumps(rec, default=str) + "\n")
            written += 1
    logger.info("dataset_sink: wrote %d records to %s", written, path)


def _stage_leaderboard_sink(s: RunState) -> None:
    """Persist the aggregate ``report`` to a JSON sidecar (A6). No-op without a report.

    Also writes the *resolved* config (R1) — the EvaluationConfig as it actually ran, with
    devices/models/effective batch size filled in — to ``<out>/config_resolved_<name>.json``,
    so a reproduction uses what ran, not the (possibly auto-completed) submitted config.
    """
    report = s.results.get("report") if s.results else None
    if not report:
        return
    from ..sinks import write_report_json

    out_dir = getattr(s.config, "output_dir", ".") if s.config else "."
    name = getattr(s.config, "experiment_name", "run") if s.config else "run"
    path = write_report_json(report, out_dir, name)
    logger.info("leaderboard_sink: report → %s", path)
    _write_resolved_config(s, out_dir, name)


def _write_resolved_config(s: RunState, out_dir: str, name: str) -> None:
    """Serialize the resolved config — the *executed DAG* — next to the report (R1).

    Node-centric YAML (graph-first Phase 5): the graph that actually ran + its non-graph
    config, no ``pipeline_mode``. Best-effort."""
    if s.config is None:
        return
    import os

    import yaml

    from ...config.graph_config import resolved_node_config

    try:
        path = os.path.join(out_dir, f"config_resolved_{name}.yaml")
        os.makedirs(out_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(
                resolved_node_config(s.config), fh,
                sort_keys=False, default_flow_style=False,
            )
        logger.info("leaderboard_sink: resolved config → %s", path)
    except Exception as exc:  # noqa: BLE001 - resolved config is a best-effort sidecar
        logger.warning("could not write resolved config: %s", exc)


def _stage_tracking_sink(s: RunState) -> None:
    """Log the aggregate ``report``'s metrics to tracking (A6). No-op without a report."""
    report = s.results.get("report") if s.results else None
    if not report:
        return
    from ..sinks import log_report_to_tracking

    log_report_to_tracking(report)
