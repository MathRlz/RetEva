"""Factory for creating LLM server instances."""

import logging
from typing import Optional

from .base import BaseLLMServer

logger = logging.getLogger(__name__)


def create_server(
    backend: str,
    model: str,
    host: str = "localhost",
    port: int = 11434,
    gpu_layers: int = -1,
    **kwargs
) -> Optional[BaseLLMServer]:
    """
    Create an LLM server instance.

    Args:
        backend: Backend type ("ollama")
        model: Model name or path
        host: Server host
        port: Server port
        gpu_layers: Number of GPU layers
        **kwargs: Additional backend-specific parameters

    Returns:
        LLM server instance or None if backend not supported
    """
    backend = backend.lower()

    if backend == "ollama":
        from .ollama import OllamaServer
        return OllamaServer(model, host, port, gpu_layers, **kwargs)

    else:
        logger.error(f"Unknown backend: {backend}. Supported: ollama")
        return None
