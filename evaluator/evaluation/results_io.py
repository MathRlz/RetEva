"""Shared run-artifact persistence.

One implementation of "save a run's outputs to `config.output_dir`" — report JSON, the
resolved-config sidecar, a run log, and per-variant `answers.jsonl`/`metrics.json` — called
from every execution path instead of a divergent copy each."""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..logging_config import add_run_file_handler
from ..config.graph_config import resolved_model_config
from ..pipeline.graph.operators import node_kind
from ..config.serialization import resolved_node_config


def get_adapter_suffix(adapter_path: Optional[str]) -> str:
    """Extract a short identifier from adapter path."""
    if not adapter_path:
        return ""
    adapter_name = os.path.basename(os.path.normpath(adapter_path))
    return f"_adapter_{adapter_name}"


def _describe_model(model_type, model_name, adapter_path) -> str:
    """Build a `type[_name][_adapter_suffix]` descriptor for one model."""
    desc = f"{model_type}"
    if model_name:
        desc += f"_{model_name}"
    desc += get_adapter_suffix(adapter_path)
    return desc


# node kind -> (model_type, model_name, adapter_path) ModelConfig field names, in the join
# order used below (asr before text).
_FAMILY_ATTRS = (
    ("asr", ("asr_model_type", "asr_model_name", "asr_adapter_path")),
    ("audio_embedding", ("audio_emb_model_type", "audio_emb_model_name", "audio_emb_adapter_path")),
    ("text_embedding", ("text_emb_model_type", "text_emb_model_name", "text_emb_adapter_path")),
)


def generate_model_description(config) -> str:
    """Generate model description string for experiment ID / output filename. Derived from
    the graph's ACTUAL model-bearing nodes (each one's resolved model, same as factory.py
    builds it), not the template label + flat default alone — a branch-only role would
    otherwise describe as its unset flat default (e.g. ``None``); a branched config now
    names every DISTINCT model it actually uses. Falls back to the flat default if the
    graph can't be built."""
    model = config.model
    try:
        graph = config.graph
    except Exception:
        graph = None
    if graph is None:
        return (
            f"{_describe_model(model.asr_model_type, model.asr_model_name, model.asr_adapter_path)}"
            f"_{_describe_model(model.text_emb_model_type, model.text_emb_model_name, model.text_emb_adapter_path)}"
        )


    parts: list = []
    for kind, attrs in _FAMILY_ATTRS:
        for n in graph.nodes:
            if node_kind(n.stage, n.params) != kind:
                continue
            mcfg = resolved_model_config(config, n)
            desc = _describe_model(*(getattr(mcfg, a) for a in attrs))
            if desc not in parts:
                parts.append(desc)
    return "_".join(parts) if parts else "no_model"


def generate_output_filename(config) -> str:
    """Generate output filename based on configuration. Includes a stable per-run id
    (``config.run_id`` if the caller stamped one — e.g. so a parent process can predict the
    exact filename a child writes; else a fresh timestamp+short-uuid) so two runs of the
    SAME ``(experiment_name, dataset, model)`` triple never collide."""
    model_desc = generate_model_description(config)
    run_id = getattr(config, "run_id", None) or (
        f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    )
    # Prefix with experiment name so the same models evaluated under different
    # configurations don't overwrite each other's results.
    exp_name = (config.experiment_name or "evaluation").strip()
    output_filename = (
        f"results_{exp_name}_{config.data.dataset_name}_{model_desc}_{run_id}.json"
    )
    output_filename = "".join(
        c for c in output_filename if c.isalnum() or c in (' ', '_', '.')
    ).rstrip()

    return output_filename


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

    ``results`` is the raw dict `run_graph`/`run_from_config` returns (flat, or
    ``{"variants": {node_id: report, ...}, ...}`` for a multi-variant graph — see
    ``executor/run.py:_collect_final_result``). ``output_path`` lets a caller pin the exact
    path (e.g. the CLI computes it once and must save to that SAME path later — calling
    `generate_output_filename` again here would mint a different filename, since it embeds
    a fresh per-call run id); computed fresh via `generate_output_filename` when omitted."""
    output_dir = config.output_dir
    os.makedirs(output_dir, exist_ok=True)

    if output_path is None:
        output_path = os.path.join(output_dir, generate_output_filename(config))
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, default=str, indent=4)
    logger.info("Results saved to: %s", output_path)

    try:
        import yaml

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
        # Single-variant run: no finalize-node id to key by, so "run" is the one variant
        # (keeps the top-level shape byte-identical for this case).
        _write_variant(os.path.join(variants_root, "run"), results, persist_intermediate)

    return output_path
