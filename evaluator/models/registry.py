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
    """Registry for model classes with plugin-style registration.

    When constructed with ``module``, the registry is *lazily populated*: the
    first lookup imports that module (whose model classes self-register via
    the ``@register_*`` decorators), so importing ``evaluator.models`` no
    longer drags every torch/transformers-heavy model module in eagerly.
    """

    def __init__(self, name: str, module: Optional[str] = None):
        """Initialize a registry.

        Args:
            name: Name of this registry (e.g., 'ASR', 'TextEmbedding')
            module: Dotted module path whose import registers this family's
                models; imported on first lookup (lazy population).
        """
        self.name = name
        self._module = module
        self._populated = module is None
        self._registry: Dict[str, Type] = {}
        self._default_names: Dict[str, str] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._params_classes: Dict[str, Type] = {}
        self._aliases: Dict[str, str] = {}  # alias -> canonical model_type

    def _ensure_populated(self) -> None:
        """Import the family module once so its decorators register models, then discover any
        third-party model plugins (entry-point group ``evaluator.models``, §5)."""
        if self._populated:
            return
        self._populated = True  # set first: registration during import re-enters
        import importlib

        assert self._module is not None
        importlib.import_module(self._module)
        try:
            from ..plugins import load_plugins

            load_plugins("evaluator.models")
        except Exception as exc:  # discovery is best-effort, never fatal
            logger.debug("model plugin discovery skipped: %s", exc)

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
            for alias in metadata.get("aliases", []) or []:
                self._aliases[alias] = model_type

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
        self._ensure_populated()
        model_type = self._aliases.get(model_type, model_type)
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
        self._ensure_populated()
        return self._default_names.get(model_type)
    
    def get_metadata(self, model_type: str) -> Dict[str, Any]:
        """Get metadata for a model type.
        
        Args:
            model_type: The model type identifier
            
        Returns:
            Metadata dictionary
        """
        self._ensure_populated()
        return self._metadata.get(model_type, {})
    
    def list_types(self) -> list[str]:
        """List all registered model types.
        
        Returns:
            List of registered model type identifiers
        """
        self._ensure_populated()
        return list(self._registry.keys())
    
    def is_registered(self, model_type: str) -> bool:
        """Check if a model type is registered.
        
        Args:
            model_type: The model type identifier
            
        Returns:
            True if registered, False otherwise
        """
        self._ensure_populated()
        return model_type in self._registry or model_type in self._aliases
    
    def get_params_class(self, model_type: str) -> Optional[Type]:
        """Return the Params dataclass for *model_type*, or None."""
        self._ensure_populated()
        return self._params_classes.get(model_type)

    def get_sizes(self, model_type: str) -> Dict[str, str]:
        """Return {size_name: model_name} for *model_type*, or empty dict."""
        self._ensure_populated()
        params_cls = self._params_classes.get(model_type)
        if params_cls is None:
            return {}
        return dict(getattr(params_cls, "SIZES", {}))

    def get_default_size(self, model_type: str) -> Optional[str]:
        """Return default size value from Params, or None."""
        self._ensure_populated()
        params_cls = self._params_classes.get(model_type)
        if params_cls is None:
            return None
        for f in dc_fields(params_cls):
            if f.name == "size":
                return f.default if f.default is not f.default_factory else None  # type: ignore[arg-type]
        return None

    def get_params_schema(self, model_type: str) -> Dict[str, Any]:
        """Return JSON-friendly schema for frontend rendering.

        The model author declares the interesting params on the inner ``Params``
        dataclass — field defaults, ``SIZES`` (size name → checkpoint, e.g. whisper's
        tiny/base/small/medium/large-v3), and an optional
        ``CHOICES: ClassVar[Dict[str, List]]`` enumerating valid values per field
        (e.g. ``{"pooling": ["mean", "cls"]}``). The builder renders exactly this
        after a model is chosen — nothing model-specific is hardcoded in the UI."""
        self._ensure_populated()
        params_cls = self._params_classes.get(model_type)
        if params_cls is None:
            return {}
        schema: Dict[str, Any] = {}
        sizes = self.get_sizes(model_type)
        choices_map = getattr(params_cls, "CHOICES", {}) or {}
        for f in dc_fields(params_cls):
            if f.name in ("SIZES", "CHOICES"):
                continue
            entry: Dict[str, Any] = {"default": f.default}
            if f.name == "size" and sizes:
                entry["choices"] = sorted(sizes.keys())
            elif f.name in choices_map:
                entry["choices"] = list(choices_map[f.name])
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


# Create global registries for each model type. Each registry lazily imports
# its family module on first lookup — the import runs the @register_* decorators.
asr_registry = ModelRegistry("ASR", module="evaluator.models.asr")
text_embedding_registry = ModelRegistry("TextEmbedding", module="evaluator.models.t2e")
audio_embedding_registry = ModelRegistry("AudioEmbedding", module="evaluator.models.a2e")
reranker_registry = ModelRegistry("Reranker", module="evaluator.models.retrieval.rag.reranker")
tts_registry = ModelRegistry("TTS", module="evaluator.models.tts")


def _make_register(registry: "ModelRegistry", label: str):
    """Build a ``register_<family>_model`` decorator bound to ``registry``."""
    def register(model_type: str, default_name: Optional[str] = None, **metadata):
        return registry.decorator(model_type, default_name, **metadata)

    register.__name__ = f"register_{label}_model"
    register.__doc__ = (
        f"Decorator for registering {label} models.\n\n"
        f"    Example:\n"
        f"        @register_{label}_model('my_type', default_name='...')\n"
        f"        class MyModel(...):\n"
        f"            ...\n"
    )
    return register


register_asr_model = _make_register(asr_registry, "asr")
register_text_embedding_model = _make_register(text_embedding_registry, "text_embedding")
register_audio_embedding_model = _make_register(audio_embedding_registry, "audio_embedding")
register_reranker_model = _make_register(reranker_registry, "reranker")
register_tts_model = _make_register(tts_registry, "tts")


# Family -> registry map. Single source for "all model families"; iterate this
# rather than hardcoding family lists elsewhere.
FAMILY_REGISTRIES: Dict[str, ModelRegistry] = {
    "asr": asr_registry,
    "text_embedding": text_embedding_registry,
    "audio_embedding": audio_embedding_registry,
    "reranker": reranker_registry,
    "tts": tts_registry,
}


def get_all_registered_models() -> Dict[str, list[str]]:
    """Get all registered models across all registries.

    Returns:
        Dictionary mapping family name to list of registered model types.
    """
    return {family: reg.list_types() for family, reg in FAMILY_REGISTRIES.items()}
