"""CLI utility functions."""

import os
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


def generate_model_description(config) -> str:
    """Generate model description string for experiment ID.

    Args:
        config: EvaluationConfig object.

    Returns:
        Model description string.
    """
    model = config.model
    if model.pipeline_mode == "audio_emb_retrieval":
        return _describe_model(
            model.audio_emb_model_type,
            model.audio_emb_model_name,
            model.audio_emb_adapter_path,
        )
    if model.pipeline_mode == "asr_only":
        return _describe_model(
            model.asr_model_type, model.asr_model_name, model.asr_adapter_path
        )
    # asr_text_retrieval
    asr_desc = _describe_model(
        model.asr_model_type, model.asr_model_name, model.asr_adapter_path
    )
    emb_desc = _describe_model(
        model.text_emb_model_type,
        model.text_emb_model_name,
        model.text_emb_adapter_path,
    )
    return f"{asr_desc}_{emb_desc}"


def generate_output_filename(config) -> str:
    """Generate output filename based on configuration.

    Args:
        config: EvaluationConfig object.

    Returns:
        Sanitized output filename for results JSON.
    """
    model_desc = generate_model_description(config)
    # Prefix with experiment name so the same models evaluated under different
    # configurations don't overwrite each other's results.
    exp_name = (config.experiment_name or "evaluation").strip()
    output_filename = (
        f"results_{exp_name}_{config.data.dataset_name}_{model_desc}.json"
    )
    # Sanitize filename
    output_filename = "".join(
        c for c in output_filename if c.isalnum() or c in (' ', '_', '.')
    ).rstrip()

    return output_filename
