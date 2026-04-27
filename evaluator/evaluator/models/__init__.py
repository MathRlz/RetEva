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

# ASR models
from .asr import WhisperModel, Wav2Vec2Model, FasterWhisperModel

# Text embedding models (T2E)
from .t2e import JinaV4Model, ClipModel, LabseModel, NemotronModel, BgeM3Model

# Audio embedding models (A2E)
from .a2e import (
    AttentionPoolAudioModel,
    AttentionPooling,
    ProjectionHead,
    MultimodalClapStyleModel,
    HuBERTAudioModel,
    WavLMAudioModel,
)

# TTS models
from .tts import PiperTTS, XTTSv2TTS, MMSTTS

# Factory functions
from .factory import (
    create_asr_model,
    create_text_embedding_model,
    create_audio_embedding_model,
    create_reranker
)

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
    
    # Text embedding models
    "JinaV4Model",
    "ClipModel",
    "LabseModel",
    "NemotronModel",
    "BgeM3Model",
    
    # Audio embedding models
    "AttentionPoolAudioModel",
    "AttentionPooling",
    "ProjectionHead",
    "MultimodalClapStyleModel",
    "HuBERTAudioModel",
    "WavLMAudioModel",
    "PiperTTS",
    "XTTSv2TTS",
    "MMSTTS",
    
    # Factory functions
    "create_asr_model",
    "create_text_embedding_model",
    "create_audio_embedding_model",
    "create_reranker",
]
