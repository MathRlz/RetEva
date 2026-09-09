"""Query optimization configuration."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .llm_backend import LLMBackendMixin


@dataclass
class QueryOptimizationConfig(LLMBackendMixin):
    """Configuration for advanced query optimization: query rewriting, HyDE, decomposition,
    and multi-query generation using LLMs to improve retrieval."""
    enabled: bool = False
    method: str = "rewrite"  # rewrite | hyde | decompose | multi_query
    model: str = "gpt-4o-mini"
    api_base: str = "https://api.openai.com/v1/chat/completions"
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.3  # 0.0-2.0
    max_iterations: int = 2
    use_initial_context: bool = True
    context_top_k: int = 3
    combine_strategy: str = "rrf"  # rrf | weighted | union | intersection
    timeout_s: int = 30
    use_local_server: bool = False
    local_server_url: Optional[str] = None
    # Sampling seed forwarded to the LLM request (inherited from top-level `llm:` unless
    # set); None = omit. temperature 0 alone does not pin sampling.
    seed: Optional[int] = None
    # Requests kept in flight for this component (inherited from top-level `llm:` unless set);
    # 1 = serial.
    concurrency: int = 1

    def __post_init__(self) -> None:
        valid_methods = {"rewrite", "hyde", "decompose", "multi_query"}
        if self.method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}, got {self.method!r}")

        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(f"temperature must be in [0, 2], got {self.temperature}")

        if self.max_iterations < 1:
            raise ValueError(f"max_iterations must be >= 1, got {self.max_iterations}")
