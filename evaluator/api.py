"""High-level convenience API for audio-to-text evaluation.

This module provides simple entry points for common evaluation tasks,
wrapping the complex pipeline creation and evaluation logic.

Public Functions
----------------
evaluate_from_config(config_path, auto_devices=True)
    Run evaluation from a YAML configuration file. Loads config, creates
    pipelines, and executes the full evaluation workflow.
    
evaluate_from_preset(preset_name, data_path=None, corpus_path=None, **overrides)
    Run evaluation using a named preset (e.g., "whisper_labse") with optional
    parameter overrides. Presets provide optimized configurations for common
    evaluation scenarios.
    
quick_evaluate(audio_dir, model="whisper", embedding="labse", ...)
    Quick one-line evaluation with minimal configuration. Ideal for rapid
    testing and prototyping with sensible defaults.

Return Types
------------
All functions return EvaluationResults by default, which provides:
- Structured access to metrics via .metrics dict or .get_metric()
- Configuration via .config attribute
- Metadata (timestamps, duration, etc.) via .metadata dict
- Pretty printing with print(results) or results.summary()
- Serialization via .save() and .load()
- Serialization via .to_dict() when needed explicitly.

Examples
--------
Basic usage with config file::

    from evaluator import evaluate_from_config
    
    results = evaluate_from_config("configs/whisper_eval.yaml")
    print(results)  # Pretty printed summary
    print(f"MRR: {results.get_metric('MRR'):.4f}")
    results.save("results/my_eval.json")

Using presets for common scenarios::

    from evaluator import evaluate_from_preset
    
    results = evaluate_from_preset(
        "whisper_labse",
        data_path="questions.json",
        corpus_path="corpus.json",
        data_batch_size=16  # Override batch size
    )
    print(results.summary())  # One-line summary

Quick evaluation for rapid testing::

    from evaluator import quick_evaluate
    
    results = quick_evaluate(
        "test_audio/",
        model="whisper",
        embedding="labse",
        k=10
    )
    print(f"Top metrics: {results.summary()}")

Error Handling
--------------
All functions raise:
- ConfigurationError: For invalid configurations or missing files
- EvaluationError: For failures during evaluation execution

Example error handling::

    from evaluator import evaluate_from_config, ConfigurationError, EvaluationError
    
    try:
        results = evaluate_from_config("config.yaml")
    except ConfigurationError as e:
        print(f"Config error: {e}")
    except EvaluationError as e:
        print(f"Evaluation failed: {e}")
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Sequence, Optional

import yaml

from .config import EvaluationConfig
from .errors import ConfigurationError, EvaluatorError
from .config.model_presets import get_preset, list_presets
from .evaluation.results import EvaluationResults
from .services import (
    run_evaluation as _service_run_evaluation,
    run_evaluation_matrix as _service_run_evaluation_matrix,
)


class EvaluationError(EvaluatorError):
    """Raised when evaluation fails."""
    pass


def evaluate_from_config(
    config_path: str,
    auto_devices: bool = True,
) -> EvaluationResults:
    """Run evaluation from a YAML configuration file.
    
    Loads the config, auto-configures devices if needed, creates pipelines,
    and runs the full evaluation.
    
    Args:
        config_path: Path to YAML configuration file.
        auto_devices: If True, auto-configure device assignments based on
            available hardware. Defaults to True.
    Returns:
        EvaluationResults containing evaluation metrics like:
        - pipeline_mode: The evaluation mode used
        - WER, CER: ASR metrics (if ASR mode)
        - MRR, MAP, Recall@k, NDCG@k: IR metrics (if retrieval mode)
        
    Raises:
        ConfigurationError: If config file doesn't exist or is invalid.
        EvaluationError: If evaluation fails during execution.
    
    Examples:
        Loading and running evaluation from a config file::
        
            >>> results = evaluate_from_config("configs/whisper_eval.yaml")
            >>> print(f"MRR: {results.metrics['MRR']:.4f}")
            MRR: 0.7523
            >>> # Or access directly
            >>> print(f"MRR: {results.get_metric('MRR'):.4f}")
            MRR: 0.7523
        
        Running on CPU by disabling auto device detection::
        
            >>> # Config file should have device settings for CPU
            >>> results = evaluate_from_config(
            ...     "configs/cpu_eval.yaml",
            ...     auto_devices=False
            ... )
        
        Using the results object::
        
            >>> results = evaluate_from_config("configs/full_eval.yaml")
            >>> print(results)  # Pretty printed output
            >>> print(results.summary())  # One-line summary
            >>> results.save("results/my_eval.json")  # Save to file
        
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise ConfigurationError(f"Config file not found: {config_path}")
    
    if not config_path.suffix in ('.yaml', '.yml'):
        raise ConfigurationError(
            f"Config file must be YAML (.yaml or .yml), got: {config_path.suffix}"
        )
    
    try:
        config = EvaluationConfig.from_yaml(str(config_path))
    except ConfigurationError:
        raise
    except (yaml.YAMLError, OSError, ValueError, TypeError) as e:
        raise ConfigurationError(
            f"Failed to parse config file: {e}. "
            "Tip: validate YAML syntax and required fields (model/data/vector_db)."
        ) from e
    
    if auto_devices:
        config = config.with_auto_devices()
    
    return run_evaluation(config)


def evaluate_from_preset(
    preset_name: str,
    data_path: Optional[str] = None,
    corpus_path: Optional[str] = None,
    **overrides: Any
) -> EvaluationResults:
    """Run evaluation using a named preset with optional overrides.
    
    Presets provide pre-configured model combinations optimized for common
    evaluation scenarios.
    
    Available presets:
        - whisper_labse: Whisper ASR + LaBSE embedding
        - wav2vec_jina: Wav2Vec2 ASR + Jina V4 embedding
        - audio_only: Direct audio embedding (no ASR)
        - fast_dev: Quick development testing (smaller models)
    
    Args:
        preset_name: Name of the preset to use.
        data_path: Optional path to questions/queries file. Overrides
            preset's data.questions_path.
        corpus_path: Optional path to corpus file. Overrides
            preset's data.corpus_path.
        **overrides: Additional config overrides using underscore notation.
            E.g., model_asr_device='cpu', data_batch_size=16.
    
    Returns:
        EvaluationResults containing evaluation metrics.
        
    Raises:
        ConfigurationError: If preset doesn't exist or overrides are invalid.
        EvaluationError: If evaluation fails during execution.
    
    Examples:
        Using a preset with custom data paths::
        
            >>> results = evaluate_from_preset(
            ...     "whisper_labse",
            ...     data_path="my_questions.json",
            ...     corpus_path="my_corpus.json"
            ... )
            >>> print(f"MRR: {results.get_metric('MRR'):.4f}")
            MRR: 0.7234
        
        Overriding batch size and retrieval settings::
        
            >>> results = evaluate_from_preset(
            ...     "fast_dev",
            ...     data_path="test_data.json",
            ...     data_batch_size=16,
            ...     vector_db_k=10
            ... )
        
        Running on CPU with trace limit::
        
            >>> results = evaluate_from_preset(
            ...     "whisper_labse",
            ...     data_path="questions.json",
            ...     model_asr_device="cpu",
            ...     model_text_emb_device="cpu",
            ...     data_trace_limit=100  # Only process first 100 samples
            ... )
        
        Listing available presets::
        
            >>> from evaluator.config.model_presets import list_presets
            >>> print(list_presets())
            ['whisper_labse', 'wav2vec_jina', 'audio_only', 'fast_dev']
        
    """
    available = list_presets()
    if preset_name not in available:
        raise ConfigurationError(
            f"Unknown preset '{preset_name}'. "
            f"Available presets: {', '.join(available)}"
        )
    
    # Build overrides for data paths
    if data_path is not None:
        overrides['data_questions_path'] = data_path
    if corpus_path is not None:
        overrides['data_corpus_path'] = corpus_path
    
    try:
        config = EvaluationConfig.from_preset(preset_name, **overrides)
    except ConfigurationError:
        raise
    except (ValueError, TypeError, KeyError) as e:
        raise ConfigurationError(
            f"Failed to create config from preset: {e}. "
            "Tip: check override names (e.g., data_batch_size, vector_db_k)."
        ) from e
    
    return run_evaluation(config)


def quick_evaluate(
    audio_dir: str,
    model: str = "whisper",
    model_size: Optional[str] = None,
    embedding: str = "labse",
    embedding_size: Optional[str] = None,
    corpus_path: Optional[str] = None,
    k: int = 5,
    batch_size: int = 32,
    trace_limit: int = 0,
    **kwargs: Any
) -> EvaluationResults:
    """Run a quick evaluation with minimal configuration.
    
    Designed for rapid testing and prototyping. Automatically configures
    devices and uses sensible defaults.
    
    Args:
        audio_dir: Directory containing audio files or path to prepared dataset.
        model: ASR model type. Options: "whisper", "wav2vec2". Default: "whisper".
        embedding: Text embedding model. Options: "labse", "jina_v4", "bge_m3".
            Default: "labse".
        corpus_path: Optional path to corpus file for retrieval.
        k: Number of retrieval results. Default: 5.
        batch_size: Processing batch size. Default: 32.
        trace_limit: Limit number of samples (0 = no limit). Default: 0.
        **kwargs: Additional overrides passed to config.
    
    Returns:
        EvaluationResults containing evaluation metrics.
        
    Raises:
        ConfigurationError: If audio_dir doesn't exist or model is invalid.
        EvaluationError: If evaluation fails during execution.
    
    Examples:
        Minimal usage with defaults::
        
            >>> results = quick_evaluate("test_audio/")
            >>> print(f"MRR: {results.get_metric('MRR'):.4f}")
            MRR: 0.6789
            >>> print(results.summary())
            MRR: 0.6789, WER: 15.23%
        
        Specifying model and embedding::
        
            >>> results = quick_evaluate(
            ...     audio_dir="prepared_data/",
            ...     model="whisper",
            ...     embedding="labse",
            ...     k=10
            ... )
        
        Quick test with limited samples::
        
            >>> results = quick_evaluate(
            ...     "test_audio/",
            ...     trace_limit=50,  # Only process first 50 samples
            ...     batch_size=16
            ... )
            >>> print(results)  # Pretty printed output
        
        With corpus for retrieval::
        
            >>> results = quick_evaluate(
            ...     audio_dir="questions/",
            ...     corpus_path="corpus.json",
            ...     model="wav2vec2",
            ...     embedding="jina_v4"
            ... )
        
    """
    audio_path = Path(audio_dir)
    if not audio_path.exists():
        raise ConfigurationError(f"Audio directory not found: {audio_dir}")
    
    # Normalize aliases to registry model_type keys
    asr_aliases = {
        "whisper": "whisper",
        "wav2vec2": "wav2vec2",
        "wav2vec": "wav2vec2",
        "faster_whisper": "faster_whisper",
    }
    emb_aliases = {
        "labse": "labse",
        "jina": "jina_v4",
        "jina_v4": "jina_v4",
        "bge": "bge_m3",
        "bge_m3": "bge_m3",
        "clip": "clip",
        "nemotron": "nemotron",
    }

    asr_type = asr_aliases.get(model.lower())
    if asr_type is None:
        raise ConfigurationError(
            f"Unknown ASR model '{model}'. "
            f"Available: {', '.join(asr_aliases.keys())}"
        )

    emb_type = emb_aliases.get(embedding.lower())
    if emb_type is None:
        raise ConfigurationError(
            f"Unknown embedding model '{embedding}'. "
            f"Available: {', '.join(emb_aliases.keys())}"
        )

    # Build config — size-based, no hardcoded HF names
    model_section: Dict[str, Any] = {
        "pipeline_mode": "asr_text_retrieval",
        "asr_model_type": asr_type,
        "text_emb_model_type": emb_type,
    }
    if model_size is not None:
        model_section["asr_size"] = model_size
    if embedding_size is not None:
        model_section["text_emb_size"] = embedding_size

    config_dict: Dict[str, Any] = {
        "experiment_name": f"quick_eval_{model}_{embedding}",
        "model": model_section,
        "data": {
            "batch_size": batch_size,
            "trace_limit": trace_limit,
            "prepared_dataset_dir": str(audio_path) if audio_path.is_dir() else None,
            "questions_path": str(audio_path) if audio_path.is_file() else None,
        },
        "vector_db": {
            "type": "inmemory",
            "k": k,
            "retrieval_mode": "dense",
        },
        "cache": {
            "enabled": True,
        },
    }
    
    if corpus_path:
        config_dict["data"]["corpus_path"] = corpus_path
    
    # Apply any additional overrides
    for key, value in kwargs.items():
        parts = key.split('_', 1)
        if len(parts) == 2 and parts[0] in ('model', 'data', 'cache', 'vector'):
            section = parts[0]
            if section == 'vector':
                section = 'vector_db'
                sub_key = parts[1].replace('db_', '', 1) if parts[1].startswith('db_') else parts[1]
            else:
                sub_key = parts[1]
            config_dict.setdefault(section, {})[sub_key] = value
        else:
            config_dict[key] = value
    
    try:
        config = EvaluationConfig.from_dict(config_dict)
        config = config.with_auto_devices()
    except ConfigurationError:
        raise
    except (ValueError, TypeError, KeyError, OSError) as e:
        raise ConfigurationError(
            f"Failed to create config: {e}. "
            "Tip: verify model/data/vector_db overrides and audio/corpus paths."
        ) from e
    
    return run_evaluation(config)


def run_evaluation(config: EvaluationConfig) -> EvaluationResults:
    """Run evaluation using a prepared configuration.
    
    This is the lowest-level stable API entrypoint for users who construct
    `EvaluationConfig` directly and want explicit control over execution.
    """
    try:
        return _service_run_evaluation(config)
    except RuntimeError as e:
        raise EvaluationError(str(e)) from e


def run_evaluation_matrix(
    base_config: EvaluationConfig,
    test_setups: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Run multiple setup variants over one base config/dataset."""
    try:
        return _service_run_evaluation_matrix(base_config, test_setups)
    except RuntimeError as e:
        raise EvaluationError(str(e)) from e
