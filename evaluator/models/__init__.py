"""Models package for evaluation framework.

This package contains all model implementations organized by category:
- ASR models: Automatic Speech Recognition models
- Text embedding models: Models for encoding text into embeddings
- Audio embedding models: Models for encoding audio directly into embeddings
- Reranker models: Models for reranking retrieval results
"""

# Base classes
from .base import ASRModel, TextEmbeddingModel, AudioEmbeddingModel

# Registry system
from .registry import (
    asr_registry,
    text_embedding_registry,
    audio_embedding_registry,
    reranker_registry,
    register_asr_model,
    register_text_embedding_model,
    register_audio_embedding_model,
    register_reranker_model,
    get_all_registered_models
)

# Factory functions
from .factory import (
    create_asr_model,
    create_text_embedding_model,
    create_audio_embedding_model,
    create_reranker
)

# Concrete model classes are exported lazily (PEP 562): importing
# ``evaluator.models`` no longer loads every torch/transformers-heavy model
# module — the submodule is imported on first attribute access. Registry
# population is independent of this: each ModelRegistry imports its family
# module on first lookup (see .registry).
_LAZY_EXPORTS = {
    # ASR models
    "WhisperModel": ".asr",
    "Wav2Vec2Model": ".asr",
    "FasterWhisperModel": ".asr",
    "SeamlessM4TASRModel": ".asr",
    # Text embedding models (T2E)
    "JinaV4Model": ".t2e",
    "ClipModel": ".t2e",
    "LabseModel": ".t2e",
    "NemotronModel": ".t2e",
    "BgeM3Model": ".t2e",
    "SonarTextModel": ".t2e",
    # Audio embedding models (A2E)
    "AttentionPoolAudioModel": ".a2e",
    "AttentionPooling": ".a2e",
    "ProjectionHead": ".a2e",
    "MultimodalClapStyleModel": ".a2e",
    "HuBERTAudioModel": ".a2e",
    "WavLMAudioModel": ".a2e",
    "SonarSpeechModel": ".a2e",
    # TTS models
    "PiperTTS": ".tts",
    "XTTSv2TTS": ".tts",
    "MMSTTS": ".tts",
    "M4TTTS": ".tts",
    # Reranker models
    "BaseReranker": ".retrieval.rag.reranker",
    "CrossEncoderReranker": ".retrieval.rag.reranker",
}


def __getattr__(name: str):
    submodule = _LAZY_EXPORTS.get(name)
    if submodule is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(submodule, __name__), name)


def __dir__():
    return sorted(set(globals()) | set(_LAZY_EXPORTS))

__all__ = [
    # Base classes
    "ASRModel",
    "TextEmbeddingModel",
    "AudioEmbeddingModel",
    
    # Registry system
    "asr_registry",
    "text_embedding_registry",
    "audio_embedding_registry",
    "reranker_registry",
    "register_asr_model",
    "register_text_embedding_model",
    "register_audio_embedding_model",
    "register_reranker_model",
    "get_all_registered_models",
    
    # ASR models
    "WhisperModel",
    "Wav2Vec2Model",
    "FasterWhisperModel",
    "SeamlessM4TASRModel",

    # Text embedding models
    "JinaV4Model",
    "ClipModel",
    "LabseModel",
    "NemotronModel",
    "BgeM3Model",
    "SonarTextModel",

    # Audio embedding models
    "AttentionPoolAudioModel",
    "AttentionPooling",
    "ProjectionHead",
    "MultimodalClapStyleModel",
    "HuBERTAudioModel",
    "WavLMAudioModel",
    "SonarSpeechModel",
    "PiperTTS",
    "XTTSv2TTS",
    "MMSTTS",
    "M4TTTS",

    # Reranker models
    "BaseReranker",
    "CrossEncoderReranker",

    # Factory functions
    "create_asr_model",
    "create_text_embedding_model",
    "create_audio_embedding_model",
    "create_reranker",
]
