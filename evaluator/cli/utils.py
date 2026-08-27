"""CLI utility functions."""

import os
import uuid
from datetime import datetime
from typing import Optional


def get_adapter_suffix(adapter_path: Optional[str]) -> str:
    """Extract a short identifier from adapter path.

    Args:
        adapter_path: Path to adapter weights directory.

    Returns:
        Suffix string like "_adapter_<name>" or empty string if no adapter.
    """
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
# order used below (mirrors the historical asr_then_text ordering for the common templates).
_FAMILY_ATTRS = (
    ("asr", ("asr_model_type", "asr_model_name", "asr_adapter_path")),
    ("audio_embedding", ("audio_emb_model_type", "audio_emb_model_name", "audio_emb_adapter_path")),
    ("text_embedding", ("text_emb_model_type", "text_emb_model_name", "text_emb_adapter_path")),
)


def generate_model_description(config) -> str:
    """Generate model description string for experiment ID / output filename.

    Derived from the graph's ACTUAL model-bearing nodes (each one's resolved model — the flat
    default overlaid by that node's own params, same as factory.py builds it), not from the
    template label + flat default alone: a branch-only role (no top-level ``nodes.<role>``
    default) previously described as its unset flat default (e.g. ``None``); a branched config
    now names every DISTINCT model it actually uses instead of only the flat one. Falls back to
    the flat default (historical asr+text-embed shape) if the graph can't be built.

    Args:
        config: EvaluationConfig object.

    Returns:
        Model description string.
    """
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

    from evaluator.config.graph_config import resolved_model_config
    from evaluator.pipeline.graph.operators import node_kind

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
    """Generate output filename based on configuration.

    Includes a stable per-run id (``config.run_id`` if the caller stamped one — e.g. so a
    parent process can predict the exact filename a child writes; else a fresh
    timestamp+short-uuid generated here) so two runs of the SAME
    ``(experiment_name, dataset, model)`` triple never collide/get skipped as "already ran"
    (they used to produce the identical filename).

    Args:
        config: EvaluationConfig object.

    Returns:
        Sanitized output filename for results JSON.
    """
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
    # Sanitize filename
    output_filename = "".join(
        c for c in output_filename if c.isalnum() or c in (' ', '_', '.')
    ).rstrip()

    return output_filename
