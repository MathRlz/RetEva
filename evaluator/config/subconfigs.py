"""Capability sub-config field name -> dataclass.

Single source for ``from_dict`` / presets / serialization. It lives in its own leaf
module so the three of them can share it without importing each other.
"""

from typing import Any, Dict

from .answer_generation import AnswerGenerationConfig
from .audio_augmentation import AudioAugmentationConfig
from .audio_synthesis import AudioSynthesisConfig
from .embedding_fusion import EmbeddingFusionConfig
from .judge import JudgeConfig
from .query_correction import QueryCorrectionConfig
from .query_optimization import QueryOptimizationConfig
from .rag_flow import RagFlowConfig

FEATURE_SUBCONFIGS: Dict[str, Any] = {
    "audio_synthesis": AudioSynthesisConfig,
    "augmentation": AudioAugmentationConfig,
    "answer_generation": AnswerGenerationConfig,
    "judge": JudgeConfig,
    "query_optimization": QueryOptimizationConfig,
    "query_correction": QueryCorrectionConfig,
    "embedding_fusion": EmbeddingFusionConfig,
    "rag": RagFlowConfig,
}
