"""High-level convenience API for audio-to-text evaluation.

Entry points: ``evaluate_from_config`` (YAML file), ``evaluate_from_preset``
(named preset + overrides), and ``quick_evaluate`` (minimal-config prototyping).
All return :class:`EvaluationResults` and raise ``ConfigurationError`` for invalid
configuration or ``EvaluationError`` for failures during execution.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from .config import EvaluationConfig
from .errors import ConfigurationError, EvaluatorError
from .config.model_presets import list_presets
from .evaluation.results import EvaluationResults
from .services import run_evaluation as _service_run_evaluation


class EvaluationError(EvaluatorError):
    """Raised when evaluation fails."""
    pass


def evaluate_from_config(
    config_path: str,
    auto_devices: bool = True,
) -> EvaluationResults:
    """Run evaluation from a YAML configuration file.

    Args:
        config_path: Path to YAML configuration file.
        auto_devices: If True, auto-configure device assignments based on
            available hardware. Defaults to True.

    Returns:
        EvaluationResults with the run's metrics (WER/CER for ASR modes,
        MRR/MAP/Recall@k/NDCG@k for retrieval modes).

    Raises:
        ConfigurationError: If config file doesn't exist or is invalid.
        EvaluationError: If evaluation fails during execution.

    Example:
        >>> results = evaluate_from_config("configs/whisper_eval.yaml")
        >>> print(f"MRR: {results.get_metric('MRR'):.4f}")
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise ConfigurationError(f"Config file not found: {config_path}")

    if config_path.suffix not in ('.yaml', '.yml'):
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
    """
    available = list_presets()
    if preset_name not in available:
        raise ConfigurationError(
            f"Unknown preset '{preset_name}'. "
            f"Available presets: {', '.join(available)}"
        )

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
    """
    audio_path = Path(audio_dir)
    if not audio_path.exists():
        raise ConfigurationError(f"Audio directory not found: {audio_dir}")

    asr_type, emb_type = _resolve_model_shortcuts(model, embedding)
    config_dict = _build_quick_eval_config_dict(
        audio_path, asr_type, emb_type, model, embedding,
        model_size, embedding_size, k, batch_size, trace_limit, corpus_path, kwargs,
    )

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


# Convenience abbreviations accepted by quick_evaluate for model/embedding names.
_ASR_SHORTCUTS: Dict[str, str] = {"wav2vec": "wav2vec2"}
_EMB_SHORTCUTS: Dict[str, str] = {"jina": "jina_v4", "bge": "bge_m3"}


def _resolve_model_shortcuts(model: str, embedding: str) -> tuple:
    """Resolve abbreviated model/embedding names to registered types (or raise)."""
    # Importing models triggers @register_* decorators so the registries are populated.
    from .models import asr_registry, text_embedding_registry

    asr_name = model.lower()
    asr_type = _ASR_SHORTCUTS.get(asr_name) or (
        asr_name if asr_registry.is_registered(asr_name) else None
    )
    if asr_type is None:
        available = sorted(set(asr_registry.list_types()) | set(_ASR_SHORTCUTS))
        raise ConfigurationError(f"Unknown ASR model '{model}'. Available: {', '.join(available)}")

    emb_name = embedding.lower()
    emb_type = _EMB_SHORTCUTS.get(emb_name) or (
        emb_name if text_embedding_registry.is_registered(emb_name) else None
    )
    if emb_type is None:
        available = sorted(set(text_embedding_registry.list_types()) | set(_EMB_SHORTCUTS))
        raise ConfigurationError(
            f"Unknown embedding model '{embedding}'. Available: {', '.join(available)}"
        )
    return asr_type, emb_type


def _apply_quick_eval_overrides(config_dict: Dict[str, Any], kwargs: Dict[str, Any]) -> None:
    """Fold quick_evaluate **kwargs into the config dict (section_subkey notation)."""
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


def _build_quick_eval_config_dict(
    audio_path: Path, asr_type: str, emb_type: str, model: str, embedding: str,
    model_size: Optional[str], embedding_size: Optional[str],
    k: int, batch_size: int, trace_limit: int, corpus_path: Optional[str],
    kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """Assemble the EvaluationConfig dict for quick_evaluate (size-based, no HF names)."""
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
        "vector_db": {"type": "inmemory", "k": k, "retrieval_mode": "dense"},
        "cache": {"enabled": True},
    }
    if corpus_path:
        config_dict["data"]["corpus_path"] = corpus_path

    _apply_quick_eval_overrides(config_dict, kwargs)
    return config_dict


def run_evaluation(config: EvaluationConfig, progress_callback=None) -> EvaluationResults:
    """Run evaluation using a prepared configuration.

    This is the lowest-level stable API entrypoint for users who construct
    `EvaluationConfig` directly and want explicit control over execution.
    """
    try:
        return _service_run_evaluation(config, progress_callback=progress_callback)
    except RuntimeError as e:
        raise EvaluationError(str(e)) from e


