"""Shared run-artifact persistence (R4/per-variant persistence).

One implementation of "save a run's outputs to `config.output_dir`" — report JSON, the
resolved-config sidecar, a co-located run log, and per-variant `answers.jsonl`/`metrics.json`
(opt-in `retrieved.jsonl` via `config.persist_intermediate_artifacts`) — called from every
execution path (plain CLI `evaluator run`, the shared evaluation service used by the webapi/
public API, and — via the CLI subprocess it launches — the webapi job runner) instead of one
divergent copy per path.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from ..logging_config import add_run_file_handler


def _slim_report(report: Dict[str, Any]) -> Dict[str, Any]:
    """A variant's report minus its large per-query blobs (the `metrics.json` shape) —
    `report.traces.query_traces` (already written whole to `answers.jsonl`) and
    `report.judge.llm_judge.details` (per-query judge verdicts; the aggregate judge scores
    stay)."""
    slim = dict(report)
    r = slim.get("report")
    if isinstance(r, dict):
        r = dict(r)
        traces = r.get("traces")
        if isinstance(traces, dict) and "query_traces" in traces:
            r["traces"] = {k: v for k, v in traces.items() if k != "query_traces"}
        judge = r.get("judge")
        if isinstance(judge, dict) and isinstance(judge.get("llm_judge"), dict):
            judge = dict(judge)
            llm_judge = dict(judge["llm_judge"])
            llm_judge.pop("details", None)
            judge["llm_judge"] = llm_judge
            r["judge"] = judge
        slim["report"] = r
    return slim


def _query_traces_of(report: Dict[str, Any]) -> List[dict]:
    r = report.get("report")
    traces = r.get("traces") if isinstance(r, dict) else None
    qt = traces.get("query_traces") if isinstance(traces, dict) else None
    return qt if isinstance(qt, list) else []


def _write_variant(variant_dir: str, report: Dict[str, Any], persist_intermediate: bool) -> None:
    os.makedirs(variant_dir, exist_ok=True)
    with open(os.path.join(variant_dir, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(_slim_report(report), fh, default=str, indent=2)

    query_traces = _query_traces_of(report)
    with open(os.path.join(variant_dir, "answers.jsonl"), "w", encoding="utf-8") as fh:
        for entry in query_traces:
            fh.write(json.dumps(entry, default=str) + "\n")

    if persist_intermediate:
        with open(os.path.join(variant_dir, "retrieved.jsonl"), "w", encoding="utf-8") as fh:
            for entry in query_traces:
                fh.write(json.dumps(
                    {"query_id": entry.get("query_id"), "retrieved": entry.get("retrieved")},
                    default=str,
                ) + "\n")


def save_run_artifacts(
    results: Dict[str, Any], config: Any, logger: Any, *, output_path: str = None,
) -> str:
    """Persist a run's outputs to ``config.output_dir``. Returns the report JSON path.

    ``results`` is the raw dict `run_graph`/`run_evaluation_from_config` returns (flat, or
    ``{"variants": {node_id: report, ...}, ...}`` for a multi-variant graph — see
    ``executor/run.py:_collect_final_result``). ``output_path`` lets a caller pin the exact
    path (e.g. the CLI computes it once, before decorating ``config.experiment_name`` for the
    run itself, and must save to that SAME path — calling `generate_output_filename` again
    here would mint a different filename, since it embeds a fresh per-call run id); computed
    fresh via `generate_output_filename` when omitted."""
    from ..cli.utils import generate_output_filename

    output_dir = config.output_dir
    os.makedirs(output_dir, exist_ok=True)

    if output_path is None:
        output_path = os.path.join(output_dir, generate_output_filename(config))
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, default=str, indent=4)
    logger.info("Results saved to: %s", output_path)

    try:
        import yaml

        from ..config.graph_config import resolved_node_config

        base = output_path[:-5] if output_path.endswith(".json") else output_path
        resolved_path = f"{base}.config_resolved.yaml"
        with open(resolved_path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(
                resolved_node_config(config), fh,
                sort_keys=False, default_flow_style=False,
            )
        logger.info("Resolved config saved to: %s", resolved_path)
    except Exception as exc:  # noqa: BLE001 - resolved config is a best-effort sidecar
        logger.warning("could not write resolved config: %s", exc)

    add_run_file_handler(output_dir, config.experiment_name)

    persist_intermediate = bool(getattr(config, "persist_intermediate_artifacts", False))
    variants_root = os.path.join(output_dir, "variants")
    variants = results.get("variants")
    if isinstance(variants, dict) and variants:
        for variant_id, report in variants.items():
            if isinstance(report, dict):
                _write_variant(
                    os.path.join(variants_root, str(variant_id)), report, persist_intermediate,
                )
    else:
        # Single-variant run: no finalize-node id to key by (Phase A keeps the top-level
        # shape flat/byte-identical for this case) — "run" is the one variant.
        _write_variant(os.path.join(variants_root, "run"), results, persist_intermediate)

    return output_path
