"""Configuration presets for common evaluation scenarios.

This module provides pre-configured settings for various evaluation scenarios,
making it easy to run experiments with optimized configurations.
"""

from typing import Dict, Any

from .embedding_fusion import EmbeddingFusionConfig
from .evaluation import EvaluationConfig
from .judge import JudgeConfig
from .query_optimization import QueryOptimizationConfig
from .vector_db import VectorDBConfig


# ===========================
# Embedding Fusion Presets
# ===========================

FUSION_BALANCED = {
    "enabled": True,
    "audio_weight": 0.5,
    "text_weight": 0.5,
    "fusion_method": "weighted",
    "normalize_before_fusion": True,
}

FUSION_AUDIO_FOCUS = {
    "enabled": True,
    "audio_weight": 0.7,
    "text_weight": 0.3,
    "fusion_method": "weighted",
    "normalize_before_fusion": True,
}

FUSION_TEXT_FOCUS = {
    "enabled": True,
    "audio_weight": 0.3,
    "text_weight": 0.7,
    "fusion_method": "weighted",
    "normalize_before_fusion": True,
}

FUSION_CONCATENATE_PCA = {
    "enabled": True,
    "audio_weight": 0.5,
    "text_weight": 0.5,
    "fusion_method": "concatenate",
    "normalize_before_fusion": True,
    "dimension_reduction": "pca",
    "target_dim": 1024,
}


# ===========================
# Query Optimization Presets
# ===========================

QUERY_REWRITE_ENABLED = {
    "enabled": True,
    "method": "rewrite",
    "model": "gpt-4o-mini",
    "temperature": 0.3,
    "max_iterations": 2,
    "use_initial_context": True,
    "context_top_k": 3,
}

QUERY_HYDE = {
    "enabled": True,
    "method": "hyde",
    "model": "gpt-4o-mini",
    "temperature": 0.7,
    "max_iterations": 1,
}

QUERY_DECOMPOSE = {
    "enabled": True,
    "method": "decompose",
    "model": "gpt-4o-mini",
    "temperature": 0.3,
    "combine_strategy": "rrf",
}

QUERY_MULTI_QUERY = {
    "enabled": True,
    "method": "multi_query",
    "model": "gpt-4o-mini",
    "temperature": 0.5,
    "combine_strategy": "rrf",
}


# ===========================
# Hybrid Retrieval Presets
# ===========================

HYBRID_BALANCED = {
    "retrieval_mode": "hybrid",
    "hybrid_dense_weight": 0.5,
    "hybrid_fusion_method": "weighted",
    "bm25_k1": 1.5,
    "bm25_b": 0.75,
}

HYBRID_SEMANTIC_FOCUS = {
    "retrieval_mode": "hybrid",
    "hybrid_dense_weight": 0.7,
    "hybrid_fusion_method": "weighted",
    "bm25_k1": 1.5,
    "bm25_b": 0.75,
}

HYBRID_KEYWORD_FOCUS = {
    "retrieval_mode": "hybrid",
    "hybrid_dense_weight": 0.3,
    "hybrid_fusion_method": "weighted",
    "bm25_k1": 1.5,
    "bm25_b": 0.75,
}

HYBRID_RRF = {
    "retrieval_mode": "hybrid",
    "hybrid_dense_weight": 0.5,
    "hybrid_fusion_method": "rrf",
    "rrf_k": 60,
    "bm25_k1": 1.5,
    "bm25_b": 0.75,
}

# ===========================
# Advanced RAG Presets
# ===========================

ADVANCED_RAG_DIVERSITY = {
    "use_mmr": True,
    "mmr_lambda": 0.6,
    "diversity_penalty": 0.3,
}


# ===========================
# Full Stack Presets
# ===========================

MEDICAL_OPTIMIZED = {
    "experiment_name": "medical_optimized",
    "embedding_fusion": EmbeddingFusionConfig(**FUSION_BALANCED),
    "query_optimization": QueryOptimizationConfig(**QUERY_REWRITE_ENABLED),
    "vector_db": VectorDBConfig(
        retrieval_mode="hybrid",
        hybrid_dense_weight=0.6,
        hybrid_fusion_method="rrf",
        reranker_enabled=True,
        use_mmr=True,
        mmr_lambda=0.7,
    ),
}

FULL_RAG_ADVANCED = {
    "experiment_name": "full_rag_advanced",
    "embedding_fusion": EmbeddingFusionConfig(**FUSION_BALANCED),
    "query_optimization": QueryOptimizationConfig(
        enabled=True,
        method="decompose",
        combine_strategy="rrf",
    ),
    "vector_db": VectorDBConfig(
        retrieval_mode="hybrid",
        hybrid_dense_weight=0.5,
        hybrid_fusion_method="rrf",
        reranker_enabled=True,
        reranker_top_k=50,
        use_mmr=True,
        mmr_lambda=0.6,
        diversity_penalty=0.2,
    ),
    "judge": JudgeConfig(
        enabled=True,
        judge_aspects=["relevance", "accuracy", "completeness"],
        score_aggregation="weighted",
        aspect_weights={"relevance": 0.5, "accuracy": 0.3, "completeness": 0.2},
        chain_of_thought=True,
    ),
}

FAST_BASELINE = {
    "experiment_name": "fast_baseline",
    "embedding_fusion": EmbeddingFusionConfig(enabled=False),
    "query_optimization": QueryOptimizationConfig(enabled=False),
    "vector_db": VectorDBConfig(
        retrieval_mode="dense",
        k=5,
    ),
}

QUALITY_FOCUSED = {
    "experiment_name": "quality_focused",
    "embedding_fusion": EmbeddingFusionConfig(**FUSION_BALANCED),
    "query_optimization": QueryOptimizationConfig(
        enabled=True,
        method="rewrite",
        max_iterations=3,
        use_initial_context=True,
    ),
    "vector_db": VectorDBConfig(
        retrieval_mode="hybrid",
        hybrid_dense_weight=0.6,
        hybrid_fusion_method="rrf",
        reranker_enabled=True,
        reranker_top_k=100,
        use_mmr=True,
        mmr_lambda=0.7,
        k=10,
    ),
    "judge": JudgeConfig(
        enabled=True,
        judge_aspects=["relevance", "accuracy", "completeness", "clarity"],
        score_aggregation="weighted",
        aspect_weights={
            "relevance": 0.4,
            "accuracy": 0.3,
            "completeness": 0.2,
            "clarity": 0.1
        },
        output_format="score_with_reasoning",
        chain_of_thought=True,
    ),
}


# ===========================
# Preset Registry
# ===========================

PRESET_REGISTRY: Dict[str, Dict[str, Any]] = {
    # Fusion presets
    "fusion_balanced": {"embedding_fusion": EmbeddingFusionConfig(**FUSION_BALANCED)},
    "fusion_audio_focus": {"embedding_fusion": EmbeddingFusionConfig(**FUSION_AUDIO_FOCUS)},
    "fusion_text_focus": {"embedding_fusion": EmbeddingFusionConfig(**FUSION_TEXT_FOCUS)},
    "fusion_concatenate_pca": {"embedding_fusion": EmbeddingFusionConfig(**FUSION_CONCATENATE_PCA)},
    
    # Query optimization presets
    "query_rewrite": {"query_optimization": QueryOptimizationConfig(**QUERY_REWRITE_ENABLED)},
    "query_hyde": {"query_optimization": QueryOptimizationConfig(**QUERY_HYDE)},
    "query_decompose": {"query_optimization": QueryOptimizationConfig(**QUERY_DECOMPOSE)},
    "query_multi": {"query_optimization": QueryOptimizationConfig(**QUERY_MULTI_QUERY)},
    
    # Hybrid retrieval presets
    "hybrid_balanced": {"vector_db": VectorDBConfig(**HYBRID_BALANCED)},
    "hybrid_semantic": {"vector_db": VectorDBConfig(**HYBRID_SEMANTIC_FOCUS)},
    "hybrid_keyword": {"vector_db": VectorDBConfig(**HYBRID_KEYWORD_FOCUS)},
    "hybrid_rrf": {"vector_db": VectorDBConfig(**HYBRID_RRF)},

    # Full stack presets
    "medical_optimized": MEDICAL_OPTIMIZED,
    "full_rag_advanced": FULL_RAG_ADVANCED,
    "fast_baseline": FAST_BASELINE,
    "quality_focused": QUALITY_FOCUSED,
}


def get_preset(preset_name: str) -> Dict[str, Any]:
    """Get a configuration preset by name.
    
    Args:
        preset_name: Name of the preset to retrieve.
        
    Returns:
        Dictionary of configuration parameters.
        
    Raises:
        KeyError: If preset name is not found.
        
    Examples:
        >>> preset = get_preset("medical_optimized")
        >>> config = EvaluationConfig(**preset)
    """
    if preset_name not in PRESET_REGISTRY:
        available = ", ".join(PRESET_REGISTRY.keys())
        raise KeyError(
            f"Unknown preset: '{preset_name}'. "
            f"Available presets: {available}"
        )
    
    return PRESET_REGISTRY[preset_name]


def list_presets() -> Dict[str, str]:
    """List all available presets with descriptions.
    
    Returns:
        Dictionary mapping preset names to descriptions.
    """
    descriptions = {
        "fusion_balanced": "Balanced audio-text fusion (50/50)",
        "fusion_audio_focus": "Audio-focused fusion (70/30)",
        "fusion_text_focus": "Text-focused fusion (30/70)",
        "fusion_concatenate_pca": "Concatenation with PCA reduction",
        
        "query_rewrite": "Iterative query rewriting with LLM",
        "query_hyde": "HyDE: Hypothetical Document Embeddings",
        "query_decompose": "Query decomposition into sub-queries",
        "query_multi": "Multi-query generation with variations",
        
        "hybrid_balanced": "Balanced hybrid retrieval (50/50)",
        "hybrid_semantic": "Semantic-focused hybrid (70/30)",
        "hybrid_keyword": "Keyword-focused hybrid (30/70)",
        "hybrid_rrf": "Reciprocal Rank Fusion hybrid",

        "medical_optimized": "Full stack optimized for medical domain",
        "full_rag_advanced": "Hybrid RRF + reranking + MMR + LLM judge",
        "fast_baseline": "Minimal configuration for fast baseline",
        "quality_focused": "Maximum quality with all enhancements",
    }
    
    return descriptions


def apply_preset(config: EvaluationConfig, preset_name: str) -> EvaluationConfig:
    """Apply a preset to an existing configuration.
    
    Args:
        config: Base configuration to modify.
        preset_name: Name of preset to apply.
        
    Returns:
        New configuration with preset applied.
        
    Examples:
        >>> config = EvaluationConfig()
        >>> config = apply_preset(config, "hybrid_balanced")
    """
    from dataclasses import replace
    from .evaluation import _FEATURE_SUBCONFIGS

    preset = dict(get_preset(preset_name))
    # Feature sub-configs now live under config.features; route them there
    # instead of passing them as (no-longer-existent) top-level fields.
    feature_overrides = {k: preset.pop(k) for k in list(preset) if k in _FEATURE_SUBCONFIGS}
    new_features = replace(config.features, **feature_overrides) if feature_overrides else config.features
    return replace(config, features=new_features, **preset)
