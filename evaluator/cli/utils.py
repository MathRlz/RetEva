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


def generate_output_filename(config) -> str:
    """Generate output filename based on configuration.
    
    Args:
        config: EvaluationConfig object.
        
    Returns:
        Sanitized output filename for results JSON.
    """
    if config.model.pipeline_mode == "audio_emb_retrieval":
        model_desc = f"{config.model.audio_emb_model_type}"
        if config.model.audio_emb_model_name:
            model_desc += f"_{config.model.audio_emb_model_name}"
        model_desc += get_adapter_suffix(config.model.audio_emb_adapter_path)
    elif config.model.pipeline_mode == "asr_only":
        model_desc = f"{config.model.asr_model_type}"
        if config.model.asr_model_name:
            model_desc += f"_{config.model.asr_model_name}"
        model_desc += get_adapter_suffix(config.model.asr_adapter_path)
    else:  # asr_text_retrieval
        asr_desc = f"{config.model.asr_model_type}"
        if config.model.asr_model_name:
            asr_desc += f"_{config.model.asr_model_name}"
        asr_desc += get_adapter_suffix(config.model.asr_adapter_path)
        emb_desc = f"{config.model.text_emb_model_type}"
        if config.model.text_emb_model_name:
            emb_desc += f"_{config.model.text_emb_model_name}"
        emb_desc += get_adapter_suffix(config.model.text_emb_adapter_path)
        model_desc = f"{asr_desc}_{emb_desc}"
    
    output_filename = f"results_{config.data.dataset_name}_{model_desc}.json"
    # Sanitize filename
    output_filename = "".join(
        c for c in output_filename if c.isalnum() or c in (' ', '_', '.')
    ).rstrip()
    
    return output_filename


def generate_model_description(config) -> str:
    """Generate model description string for experiment ID.
    
    Args:
        config: EvaluationConfig object.
        
    Returns:
        Model description string.
    """
    if config.model.pipeline_mode == "audio_emb_retrieval":
        model_desc = f"{config.model.audio_emb_model_type}"
        if config.model.audio_emb_model_name:
            model_desc += f"_{config.model.audio_emb_model_name}"
        model_desc += get_adapter_suffix(config.model.audio_emb_adapter_path)
    elif config.model.pipeline_mode == "asr_only":
        model_desc = f"{config.model.asr_model_type}"
        if config.model.asr_model_name:
            model_desc += f"_{config.model.asr_model_name}"
        model_desc += get_adapter_suffix(config.model.asr_adapter_path)
    else:  # asr_text_retrieval
        asr_desc = f"{config.model.asr_model_type}"
        if config.model.asr_model_name:
            asr_desc += f"_{config.model.asr_model_name}"
        asr_desc += get_adapter_suffix(config.model.asr_adapter_path)
        emb_desc = f"{config.model.text_emb_model_type}"
        if config.model.text_emb_model_name:
            emb_desc += f"_{config.model.text_emb_model_name}"
        emb_desc += get_adapter_suffix(config.model.text_emb_adapter_path)
        model_desc = f"{asr_desc}_{emb_desc}"
    
    return model_desc
