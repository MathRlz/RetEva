"""Centralized model registry for all model types.

This module provides a plugin-style registry system that allows models to
self-register, making it easy to add new models without modifying factory code.

Each model class may declare an inner ``Params`` dataclass with a ``size``
field and a ``SIZES`` class-var mapping size names to HuggingFace model
identifiers.  The registry auto-discovers this at registration time and
exposes it via :meth:`ModelRegistry.get_params_class` /
:func:`resolve_model_name`.
"""
from dataclasses import fields as dc_fields, asdict as dc_asdict
from typing import Dict, List, Type, Callable, Optional, Any
import logging

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Registry for model classes with plugin-style registration."""
    
    def __init__(self, name: str):
        """Initialize a registry.
        
        Args:
            name: Name of this registry (e.g., 'ASR', 'TextEmbedding')
        """
        self.name = name
        self._registry: Dict[str, Type] = {}
        self._default_names: Dict[str, str] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._params_classes: Dict[str, Type] = {}
    
    def register(
        self,
        model_type: str,
        model_class: Type,
        default_name: Optional[str] = None,
        **metadata
    ):
        """Register a model class.
        
        Args:
            model_type: Unique identifier for this model type (e.g., 'whisper')
            model_class: The model class to register
            default_name: Default model name/checkpoint if not specified
            **metadata: Additional metadata (description, tags, etc.)
        """
        if model_type in self._registry:
            class_name = model_class.__name__ if model_class is not None else "None"
            logger.warning(
                f"Model type '{model_type}' already registered in {self.name} registry. "
                f"Overwriting with {class_name}"
            )
        
        self._registry[model_type] = model_class
        if default_name:
            self._default_names[model_type] = default_name
        if metadata:
            self._metadata[model_type] = metadata

        # Auto-detect inner Params dataclass
        params_cls = getattr(model_class, "Params", None)
        if params_cls is not None:
            self._params_classes[model_type] = params_cls

        class_name = model_class.__name__ if model_class is not None else "None"
        logger.debug(f"Registered {model_type} -> {class_name} in {self.name} registry")
    
    def get(self, model_type: str) -> Type:
        """Get a registered model class.
        
        Args:
            model_type: The model type identifier
            
        Returns:
            The registered model class
            
        Raises:
            ValueError: If model type is not registered
        """
        if model_type not in self._registry:
            available = ', '.join(self._registry.keys())
            raise ValueError(
                f"Unknown {self.name} model type: '{model_type}'. "
                f"Available types: {available}"
            )
        return self._registry[model_type]
    
    def get_default_name(self, model_type: str) -> Optional[str]:
        """Get the default model name for a model type.
        
        Args:
            model_type: The model type identifier
            
        Returns:
            Default model name or None
        """
        return self._default_names.get(model_type)
    
    def get_metadata(self, model_type: str) -> Dict[str, Any]:
        """Get metadata for a model type.
        
        Args:
            model_type: The model type identifier
            
        Returns:
            Metadata dictionary
        """
        return self._metadata.get(model_type, {})
    
    def list_types(self) -> list[str]:
        """List all registered model types.
        
        Returns:
            List of registered model type identifiers
        """
        return list(self._registry.keys())
    
    def is_registered(self, model_type: str) -> bool:
        """Check if a model type is registered.
        
        Args:
            model_type: The model type identifier
            
        Returns:
            True if registered, False otherwise
        """
        return model_type in self._registry
    
    def get_params_class(self, model_type: str) -> Optional[Type]:
        """Return the Params dataclass for *model_type*, or None."""
        return self._params_classes.get(model_type)

    def get_sizes(self, model_type: str) -> Dict[str, str]:
        """Return {size_name: model_name} for *model_type*, or empty dict."""
        params_cls = self._params_classes.get(model_type)
        if params_cls is None:
            return {}
        return dict(getattr(params_cls, "SIZES", {}))

    def get_default_size(self, model_type: str) -> Optional[str]:
        """Return default size value from Params, or None."""
        params_cls = self._params_classes.get(model_type)
        if params_cls is None:
            return None
        for f in dc_fields(params_cls):
            if f.name == "size":
                return f.default if f.default is not f.default_factory else None  # type: ignore[arg-type]
        return None

    def get_params_schema(self, model_type: str) -> Dict[str, Any]:
        """Return JSON-friendly schema for frontend rendering."""
        params_cls = self._params_classes.get(model_type)
        if params_cls is None:
            return {}
        schema: Dict[str, Any] = {}
        sizes = self.get_sizes(model_type)
        for f in dc_fields(params_cls):
            if f.name == "SIZES":
                continue
            entry: Dict[str, Any] = {"default": f.default}
            if f.name == "size" and sizes:
                entry["choices"] = sorted(sizes.keys())
            schema[f.name] = entry
        return schema

    def resolve_model_name(
        self,
        model_type: str,
        *,
        size: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> str:
        """Resolve final HF model name.

        Priority: explicit *model_name* > *size* via SIZES > registry default.
        """
        if model_name:
            return model_name
        if size is not None:
            sizes = self.get_sizes(model_type)
            if size in sizes:
                return sizes[size]
            available = ", ".join(sorted(sizes.keys())) if sizes else "(none)"
            raise ValueError(
                f"Unknown size '{size}' for {self.name} model '{model_type}'. "
                f"Available sizes: {available}"
            )
        default = self.get_default_name(model_type)
        if default:
            return default
        # Fallback: try default size from Params
        default_size = self.get_default_size(model_type)
        if default_size is not None:
            sizes = self.get_sizes(model_type)
            if default_size in sizes:
                return sizes[default_size]
        raise ValueError(
            f"No model_name, size, or default for {self.name} model '{model_type}'."
        )

    def decorator(
        self,
        model_type: str,
        default_name: Optional[str] = None,
        **metadata
    ):
        """Decorator for registering model classes.
        
        Example:
            @asr_registry.decorator('whisper', default_name='openai/whisper-medium')
            class WhisperModel(ASRModel):
                ...
        
        Args:
            model_type: Unique identifier for this model type
            default_name: Default model name/checkpoint
            **metadata: Additional metadata
        """
        def decorator_fn(cls):
            self.register(model_type, cls, default_name, **metadata)
            return cls
        return decorator_fn


# Create global registries for each model type
asr_registry = ModelRegistry("ASR")
text_embedding_registry = ModelRegistry("TextEmbedding")
audio_embedding_registry = ModelRegistry("AudioEmbedding")
reranker_registry = ModelRegistry("Reranker")


def register_asr_model(
    model_type: str,
    default_name: Optional[str] = None,
    **metadata
):
    """Decorator for registering ASR models.
    
    Example:
        @register_asr_model('whisper', default_name='openai/whisper-medium')
        class WhisperModel(ASRModel):
            ...
    """
    return asr_registry.decorator(model_type, default_name, **metadata)


def register_text_embedding_model(
    model_type: str,
    default_name: Optional[str] = None,
    **metadata
):
    """Decorator for registering text embedding models.
    
    Example:
        @register_text_embedding_model('labse', default_name='sentence-transformers/LaBSE')
        class LabseModel(TextEmbeddingModel):
            ...
    """
    return text_embedding_registry.decorator(model_type, default_name, **metadata)


def register_audio_embedding_model(
    model_type: str,
    default_name: Optional[str] = None,
    **metadata
):
    """Decorator for registering audio embedding models.
    
    Example:
        @register_audio_embedding_model('attention_pool')
        class AttentionPoolAudioModel(AudioEmbeddingModel):
            ...
    """
    return audio_embedding_registry.decorator(model_type, default_name, **metadata)


def register_reranker_model(
    model_type: str,
    default_name: Optional[str] = None,
    **metadata
):
    """Decorator for registering reranker models.
    
    Example:
        @register_reranker_model('cross_encoder', default_name='cross-encoder/ms-marco-MiniLM-L-6-v2')
        class CrossEncoderReranker(BaseReranker):
            ...
    """
    return reranker_registry.decorator(model_type, default_name, **metadata)


def get_all_registered_models() -> Dict[str, list[str]]:
    """Get all registered models across all registries.
    
    Returns:
        Dictionary mapping registry name to list of model types
    """
    return {
        "asr": asr_registry.list_types(),
        "text_embedding": text_embedding_registry.list_types(),
        "audio_embedding": audio_embedding_registry.list_types(),
        "reranker": reranker_registry.list_types(),
    }
