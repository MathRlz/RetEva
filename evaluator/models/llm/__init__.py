"""LLM Server components for the evaluator: the Ollama local-LLM backend and a curated
local-model list for the picker."""

from .ollama import OllamaServer, ServerHealth, ServerStatus
from .registry import CURATED_LOCAL_MODELS

__version__ = "0.1.0"

__all__ = [
    "OllamaServer",
    "ServerHealth",
    "ServerStatus",
    "CURATED_LOCAL_MODELS",
]
