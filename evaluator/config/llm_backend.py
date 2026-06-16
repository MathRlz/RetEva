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
    # Optional run-level cap on cumulative LLM tokens (0 = no cap). Exceeding it aborts the
    # run with BudgetExceededError so an expensive judge/answer-gen sweep can't run away (T8).
    max_tokens_budget: int = 0

    def get_api_base(self) -> str:
        if self.use_local_server:
            if not self.local_server_url:
                raise ValueError(
                    "use_local_server=True but local_server_url is not set. "
                    "Set local_server_url in your LLM config."
                )
            return self.local_server_url
        return self.api_base


class LLMBackendMixin:
    """Shared ``to_llm_config`` + ``get_api_base`` for the LLM-using feature configs.

    Methods only (no fields), so mixing it into a ``@dataclass`` doesn't disturb that config's
    field layout. The host config must declare the standard LLM fields (``model``, ``api_base``,
    ``api_key_env``, ``temperature``, ``timeout_s``, ``use_local_server``, ``local_server_url``).
    """

    def get_api_base(self) -> str:
        if self.use_local_server and self.local_server_url:
            return self.local_server_url
        return self.api_base

    def to_llm_config(self) -> "LLMConfig":
        return LLMConfig(
            model=self.model,
            api_base=self.api_base,
            api_key_env=self.api_key_env,
            temperature=self.temperature,
            timeout_s=self.timeout_s,
            use_local_server=self.use_local_server,
            local_server_url=self.local_server_url,
        )
