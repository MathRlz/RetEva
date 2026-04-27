"""Shared LLM backend configuration."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMConfig:
    """Shared LLM backend settings used by judge, answer_generation, and query_optimization.

    Set once under ``EvaluationConfig.llm``; each component inherits these values
    unless it explicitly overrides them in its own section.
    """
    model: str = "gpt-4o-mini"
    api_base: str = "https://api.openai.com/v1/chat/completions"
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.0
    timeout_s: int = 60
    use_local_server: bool = False
    local_server_url: Optional[str] = None

    def get_api_base(self) -> str:
        if self.use_local_server and self.local_server_url:
            return self.local_server_url
        return self.api_base
