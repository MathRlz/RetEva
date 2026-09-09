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
    # Sampling seed sent with each request (Ollama/vLLM honor it; OpenAI treats it as a
    # best-effort hint). None = omit the field, so a server rejecting unknown keys is
    # unaffected. temperature 0 alone does NOT pin sampling.
    seed: Optional[int] = None
    # Optional run-level cap on cumulative LLM tokens (0 = no cap). Exceeding it aborts the
    # run with BudgetExceededError so an expensive judge/answer-gen sweep can't run away.
    max_tokens_budget: int = 0
    # How many requests to keep in flight for the per-item LLM loops (answer generation,
    # judging). 1 = strictly serial, the safe default for a rate-limited remote API; a
    # local server (Ollama/vLLM) batches concurrent requests, so 4 is typically 2.5-3.5x
    # on one GPU (the SERVER must allow it too, e.g. OLLAMA_NUM_PARALLEL=4, and each slot
    # holds its own KV cache).
    #
    # Measured cost: concurrency changes batch composition and hence the numerics — two
    # serial runs matched byte-identically, serial vs concurrency=4 matched only 5/12
    # answers (decisions unchanged, wording not). Keep 1 to quote a run exactly; raise it
    # for sweeps.
    concurrency: int = 1
    # Per-call retries for TRANSIENT failures only (timeouts, connection errors, HTTP 429/5xx,
    # never auth/schema 4xx); short backoff so one hiccup doesn't drop an item. 0 disables.
    max_retries: int = 2

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
            # optional on the component configs (a component may predate the field)
            seed=getattr(self, "seed", None),
        )
