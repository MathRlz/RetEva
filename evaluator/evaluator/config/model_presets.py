"""Pre-configured evaluation presets for common scenarios.

Provides ready-to-use configurations so users don't need to specify 70+ parameters.
Device assignments use auto-detection by default to work on both GPU and CPU machines.
"""

from typing import Dict, Any


def _auto_device() -> str:
    """Get auto-detected device for presets."""
    from .base import detect_device
    return detect_device()


# Whisper ASR + LaBSE text embedding (asr_text_retrieval mode)
WHISPER_LABSE_PRESET: Dict[str, Any] = {
    "experiment_name": "whisper_labse_eval",
    "model": {
        "pipeline_mode": "asr_text_retrieval",
        "asr_model_type": "whisper",
        "asr_size": "base",
        "text_emb_model_type": "labse",
    },
    "data": {
        "batch_size": 32,
    },
    "vector_db": {
        "type": "inmemory",
        "k": 5,
        "retrieval_mode": "dense",
    },
}


# Wav2Vec2 ASR + Jina V4 text embedding
WAV2VEC_JINA_PRESET: Dict[str, Any] = {
    "experiment_name": "wav2vec_jina_eval",
    "model": {
        "pipeline_mode": "asr_text_retrieval",
        "asr_model_type": "wav2vec2",
        "asr_size": "base",
        "text_emb_model_type": "jina_v4",
    },
    "data": {
        "batch_size": 32,
    },
    "vector_db": {
        "type": "inmemory",
        "k": 5,
        "retrieval_mode": "dense",
    },
}


# Direct audio embedding mode (audio_emb_retrieval)
AUDIO_ONLY_PRESET: Dict[str, Any] = {
    "experiment_name": "audio_embedding_eval",
    "model": {
        "pipeline_mode": "audio_emb_retrieval",
        "asr_model_type": None,
        "text_emb_model_type": None,
        "audio_emb_model_type": "attention_pool",
        "audio_emb_size": "large",
        "audio_emb_dim": 2048,
        "audio_emb_dropout": 0.1,
    },
    "data": {
        "batch_size": 16,
    },
    "vector_db": {
        "type": "inmemory",
        "k": 5,
        "retrieval_mode": "dense",
    },
}


# Quick development preset with smaller models/fewer samples
FAST_DEV_PRESET: Dict[str, Any] = {
    "experiment_name": "fast_dev",
    "model": {
        "pipeline_mode": "asr_text_retrieval",
        "asr_model_type": "whisper",
        "asr_size": "tiny",
        "text_emb_model_type": "labse",
    },
    "data": {
        "batch_size": 8,
        "trace_limit": 50,
    },
    "cache": {
        "enabled": True,
    },
    "logging": {
        "console_level": "DEBUG",
    },
    "vector_db": {
        "type": "inmemory",
        "k": 5,
    },
    "checkpoint_enabled": False,
}


# Registry of all presets
_PRESETS: Dict[str, Dict[str, Any]] = {
    "whisper_labse": WHISPER_LABSE_PRESET,
    "wav2vec_jina": WAV2VEC_JINA_PRESET,
    "audio_only": AUDIO_ONLY_PRESET,
    "fast_dev": FAST_DEV_PRESET,
}


def get_preset(name: str, auto_devices: bool = True) -> Dict[str, Any]:
    """Retrieve a preset configuration by name.

    Presets provide pre-configured model combinations optimized for common
    evaluation scenarios. Each preset specifies model types, embedding models,
    and sensible defaults for batch size and retrieval settings.

    Args:
        name: Name of the preset (whisper_labse, wav2vec_jina, audio_only, fast_dev)
        auto_devices: If True (default), auto-configure device assignments based
                      on available hardware. Set to False to use ModelConfig defaults.

    Returns:
        Dictionary with preset configuration values.

    Raises:
        ValueError: If preset name is not found.

    Examples:
        Listing available presets::

            >>> from evaluator.config.model_presets import list_presets
            >>> list_presets()
            ['whisper_labse', 'wav2vec_jina', 'audio_only', 'fast_dev']

        Getting a preset configuration::

            >>> preset = get_preset("whisper_labse")
            >>> preset["model"]["asr_model_type"]
            'whisper'
            >>> preset["model"]["text_emb_model_type"]
            'labse'

        Using preset with EvaluationConfig and overrides::

            >>> from evaluator.config import EvaluationConfig
            >>> config = EvaluationConfig.from_preset(
            ...     "fast_dev",
            ...     data_batch_size=16,
            ...     model_asr_device="cpu"
            ... )
            >>> config.data.batch_size
            16

        Disabling auto device configuration::

            >>> preset = get_preset("audio_only", auto_devices=False)
            >>> # Uses default device assignments from ModelConfig
    """
    if name not in _PRESETS:
        available = ", ".join(_PRESETS.keys())
        raise ValueError(f"Unknown preset '{name}'. Available presets: {available}")

    # Return a deep copy to prevent mutation
    import copy
    preset = copy.deepcopy(_PRESETS[name])

    # Auto-configure devices if requested
    if auto_devices:
        from . import get_available_gpu_count
        gpu_count = get_available_gpu_count()

        if "model" not in preset:
            preset["model"] = {}

        if gpu_count == 0:
            preset["model"]["asr_device"] = "cpu"
            preset["model"]["text_emb_device"] = "cpu"
            preset["model"]["audio_emb_device"] = "cpu"
        elif gpu_count == 1:
            preset["model"]["asr_device"] = "cuda:0"
            preset["model"]["text_emb_device"] = "cuda:0"
            preset["model"]["audio_emb_device"] = "cuda:0"
        else:
            preset["model"]["asr_device"] = "cuda:0"
            preset["model"]["text_emb_device"] = "cuda:1"
            preset["model"]["audio_emb_device"] = "cuda:0"

    return preset


def list_presets() -> list[str]:
    """Return list of available preset names."""
    return list(_PRESETS.keys())
