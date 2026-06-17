"""Factory functions for creating model instances."""
import inspect
import logging
from typing import Optional, Callable, Dict, Any
import torch
from .registry import asr_registry, text_embedding_registry, audio_embedding_registry, reranker_registry

logger = logging.getLogger(__name__)


def _quantization_into_params(
    model_type: str, model_class, quantization: Optional[str], extra_params: Dict[str, Any]
) -> Dict[str, Any]:
    """Fold a ``quantization`` strategy into a model's constructor params when the model
    supports it — its ``__init__`` takes a ``quantization`` arg or ``**kwargs`` (Roadmap §8
    1a). Opt-in, like model params: a model that doesn't support it gets a clear warning and
    loads full precision instead of a cryptic constructor error. No-op when unset."""
    if not quantization:
        return extra_params
    try:
        sig = inspect.signature(model_class.__init__)
        accepts = "quantization" in sig.parameters or any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
    except (TypeError, ValueError):
        accepts = False
    if accepts:
        return {**extra_params, "quantization": quantization}
    logger.warning(
        "model '%s' (%s) does not support quantization=%r; loading full precision",
        model_type, model_class.__name__, quantization,
    )
    return extra_params


def _check_extra_params(model_type: str, model_class, extra_params: Dict[str, Any]) -> None:
    """Fail loud + clear when extra params don't fit the model's constructor.

    Guard for incoherent configs (e.g. a form/preset merge that mixed one model's type
    with another model's params — `whisper` + m4t's `tgt_lang`): without this the error
    surfaces as a cryptic ``__init__() got an unexpected keyword argument`` deep in a
    run. Constructors that take ``**kwargs`` accept anything and are skipped."""
    if not extra_params:
        return
    try:
        sig = inspect.signature(model_class.__init__)
    except (TypeError, ValueError):  # builtins / C extensions — can't introspect
        return
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return
    accepted = {n for n in sig.parameters if n != "self"}
    unknown = sorted(set(extra_params) - accepted)
    if unknown:
        raise ValueError(
            f"Model '{model_type}' ({model_class.__name__}) does not accept "
            f"param(s) {unknown}; it accepts {sorted(accepted)}. "
            "This usually means the config mixes one model's type with another "
            "model's name/params (e.g. a preset merged with a different form selection)."
        )


def create_asr_model(model_type: str, model_name: Optional[str] = None,
                     size: Optional[str] = None,
                     adapter_path: Optional[str] = None, device: str = "cuda:0",
                     quantization: Optional[str] = None,
                     **extra_params):
    """
    Factory function to create ASR models.

    Args:
        model_type: Type of ASR model (e.g., 'whisper', 'wav2vec2')
        model_name: Optional model name/path (wins over size)
        size: Optional size shorthand (e.g., 'large-v3') resolved via Params.SIZES
        adapter_path: Optional path to adapter weights
        device: Device to load model on
        **extra_params: Forwarded to model constructor (e.g., compute_type)

    Returns:
        ASRModel instance
    """
    model_class = asr_registry.get(model_type)
    name = asr_registry.resolve_model_name(model_type, size=size, model_name=model_name)
    extra_params = _quantization_into_params(model_type, model_class, quantization, extra_params)
    _check_extra_params(model_type, model_class, extra_params)

    return model_class(name, adapter_path=adapter_path, **extra_params).to(torch.device(device))


def create_text_embedding_model(model_type: str, model_name: Optional[str] = None,
                                size: Optional[str] = None,
                                device: str = "cuda:0",
                                quantization: Optional[str] = None, **extra_params):
    """
    Factory function to create text embedding models.

    Args:
        model_type: Type of text embedding model (e.g., 'labse', 'jina_v4', 'clip')
        model_name: Optional model name/path (wins over size)
        size: Optional size shorthand resolved via Params.SIZES
        device: Device to load model on
        **extra_params: Forwarded to model constructor

    Returns:
        TextEmbeddingModel instance or None for special cases
    """
    if model_type == "clap_text":
        return None

    model_class = text_embedding_registry.get(model_type)
    name = text_embedding_registry.resolve_model_name(model_type, size=size, model_name=model_name)
    extra_params = _quantization_into_params(model_type, model_class, quantization, extra_params)
    _check_extra_params(model_type, model_class, extra_params)

    return model_class(name, **extra_params).to(torch.device(device))


AudioBuilder = Callable[..., Any]
_AUDIO_EMBEDDING_BUILDERS: Dict[str, AudioBuilder] = {}


def register_audio_embedding_builder(model_type: str, builder: AudioBuilder) -> None:
    _AUDIO_EMBEDDING_BUILDERS[model_type] = builder


def _build_attention_pool_audio(
    model_class,
    *,
    model_type: str,
    model_name: Optional[str],
    size: Optional[str],
    model_path: Optional[str],
    emb_dim: int,
    dropout: float,
    device: str,
    **extra,
):
    _check_extra_params(model_type, model_class, extra)
    name = audio_embedding_registry.resolve_model_name(model_type, size=size, model_name=model_name)
    return model_class(
        audio_encoder_name=name,
        emb_dim=emb_dim,
        model_path=model_path,
        dropout=dropout,
        pooling=extra.get("pooling", "attention"),
    ).to(torch.device(device))


def _build_clap_style_audio(
    model_class,
    *,
    model_type: str,
    model_name: Optional[str],
    size: Optional[str],
    model_path: Optional[str],
    emb_dim: int,
    dropout: float,
    device: str,
    **extra,
):
    _check_extra_params(model_type, model_class, extra)
    if model_path is None:
        raise ValueError(
            "clap_style model requires 'model_path' to be specified.\n"
            "Provide the path to pre-trained CLAP model weights."
        )
    return model_class(
        model_path=model_path,
        device=device,
    )


register_audio_embedding_builder("attention_pool", _build_attention_pool_audio)
register_audio_embedding_builder("attention_pool_m4t", _build_attention_pool_audio)
register_audio_embedding_builder("clap_style", _build_clap_style_audio)


def create_audio_embedding_model(model_type: str, model_name: Optional[str] = None,
                                 size: Optional[str] = None,
                                 model_path: Optional[str] = None,
                                 emb_dim: int = 2048, dropout: float = 0.1,
                                 device: str = "cuda:0",
                                 quantization: Optional[str] = None, **extra_params):
    """
    Factory function to create audio embedding models.

    Args:
        model_type: Type of audio embedding model (e.g., 'attention_pool', 'clap_style')
        model_name: Optional audio encoder name/path (wins over size)
        size: Optional size shorthand resolved via Params.SIZES
        model_path: Path to pre-trained model weights
        emb_dim: Embedding dimension
        dropout: Dropout rate
        device: Device to load model on
        **extra_params: Forwarded to builder

    Returns:
        AudioEmbeddingModel instance
    """
    model_class = audio_embedding_registry.get(model_type)
    extra_params = _quantization_into_params(model_type, model_class, quantization, extra_params)
    builder = _AUDIO_EMBEDDING_BUILDERS.get(model_type)
    if builder is None:
        # Generic fallback: resolve name, construct with model_name kwarg
        name = audio_embedding_registry.resolve_model_name(model_type, size=size, model_name=model_name)
        _check_extra_params(model_type, model_class, extra_params)
        return model_class(model_name=name, **extra_params).to(torch.device(device))
    return builder(
        model_class,
        model_type=model_type,
        model_name=model_name,
        size=size,
        model_path=model_path,
        emb_dim=emb_dim,
        dropout=dropout,
        device=device,
        **extra_params,
    )


def create_reranker(
    model_type: str = "cross_encoder",
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    batch_size: int = 32,
    max_length: int = 512,
):
    """
    Factory function to create reranker models.
    
    Args:
        model_type: Type of reranker model (e.g., 'cross_encoder')
        model_name: Optional model name/path (uses registry default if None)
        device: Device to load model on (auto-detect if None)
        batch_size: Batch size for scoring
        max_length: Maximum sequence length for tokenization
        
    Returns:
        BaseReranker instance
    """
    model_class = reranker_registry.get(model_type)
    name = model_name or reranker_registry.get_default_name(model_type)
    
    if name is None:
        available = reranker_registry.list_types()
        raise ValueError(
            f"No default name for reranker type '{model_type}'. Please provide model_name.\n"
            f"Registered rerankers: {', '.join(available)}"
        )
    
    return model_class(
        model_name=name,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
    )


