"""
LLM Server components for the evaluator.

This package provides:
- Base LLM server interface
- Ollama server implementation
- Model registry for supported LLM models
- Factory functions for creating servers
"""

from .base import BaseLLMServer
from .ollama import OllamaServer
from .registry import ModelRegistry
from .factory import create_server

__version__ = "0.1.0"

__all__ = [
    "BaseLLMServer",
    "OllamaServer", 
    "ModelRegistry",
    "create_server"
]
