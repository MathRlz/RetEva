"""Sink stage handlers: dataset / leaderboard / tracking output nodes.

Moved verbatim from the former ``evaluation/phased.py`` (Phase 1, X5). Each handler
registers itself via ``@register_stage_handler`` at import time.
"""

from __future__ import annotations

from ..stage_registry import register_stage_handler
from ...logging_config import get_logger
from ..executor.state import RunState

logger = get_logger(__name__)


@register_stage_handler("dataset_sink", self_timed=True)
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


@register_stage_handler("leaderboard_sink", self_timed=True)
def _stage_leaderboard_sink(s: RunState) -> None:
    """Persist the aggregate ``report`` to a JSON sidecar (A6). No-op without a report."""
    report = s.results.get("report") if s.results else None
    if not report:
        return
    from ..sinks import write_report_json

    out_dir = getattr(s.config, "output_dir", ".") if s.config else "."
    name = getattr(s.config, "experiment_name", "run") if s.config else "run"
    path = write_report_json(report, out_dir, name)
    logger.info("leaderboard_sink: report → %s", path)


@register_stage_handler("tracking_sink", self_timed=True)
def _stage_tracking_sink(s: RunState) -> None:
    """Log the aggregate ``report``'s metrics to tracking (A6). No-op without a report."""
    report = s.results.get("report") if s.results else None
    if not report:
        return
    from ..sinks import log_report_to_tracking

    log_report_to_tracking(report)
