"""Query optimization configuration."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class QueryOptimizationConfig:
    """
    Configuration for advanced query optimization techniques.

    Enables query rewriting, HyDE (Hypothetical Document Embeddings), query
    decomposition, and multi-query generation using LLMs to improve retrieval.

    Attributes:
        enabled: Whether query optimization is enabled. Default: False.
        method: Optimization method to use. Default: "rewrite".
            Options: "rewrite", "hyde", "decompose", "multi_query".
        model: LLM model identifier. Default: "gpt-4o-mini".
        api_base: OpenAI-compatible endpoint URL.
        api_key_env: Env var name for the API key. Default: "OPENAI_API_KEY".
        temperature: Sampling temperature (0.0-2.0). Default: 0.3.
        max_iterations: Maximum refinement iterations for rewrite. Default: 2.
        use_initial_context: Use retrieval context for iterative refinement. Default: True.
        context_top_k: Number of top results to use as context. Default: 3.
        combine_strategy: How to combine multi-query results. Default: "rrf".
            Options: "rrf", "weighted", "union", "intersection".
        timeout_s: Request timeout in seconds. Default: 30.
        use_local_server: Use local_server_url instead of api_base. Default: False.
        local_server_url: URL for local LLM server (e.g. Ollama). Default: None.

    Examples:
        >>> config = QueryOptimizationConfig(enabled=True, method="rewrite")
        >>> config = QueryOptimizationConfig(enabled=True, method="hyde", temperature=0.7)
        >>> config = QueryOptimizationConfig(
        ...     enabled=True, method="decompose", combine_strategy="weighted"
        ... )
    """
    enabled: bool = False
    method: str = "rewrite"  # rewrite | hyde | decompose | multi_query
    model: str = "gpt-4o-mini"
    api_base: str = "https://api.openai.com/v1/chat/completions"
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.3
    max_iterations: int = 2
    use_initial_context: bool = True
    context_top_k: int = 3
    combine_strategy: str = "rrf"  # rrf | weighted | union | intersection
    timeout_s: int = 30
    use_local_server: bool = False
    local_server_url: Optional[str] = None

    def __post_init__(self) -> None:
        valid_methods = {"rewrite", "hyde", "decompose", "multi_query"}
        if self.method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}, got {self.method!r}")

        valid_strategies = {"rrf", "weighted", "union", "intersection"}
        if self.combine_strategy not in valid_strategies:
            raise ValueError(
                f"combine_strategy must be one of {valid_strategies}, got {self.combine_strategy!r}"
            )

        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(f"temperature must be in [0, 2], got {self.temperature}")

        if self.max_iterations < 1:
            raise ValueError(f"max_iterations must be >= 1, got {self.max_iterations}")

    def get_api_base(self) -> str:
        if self.use_local_server and self.local_server_url:
            return self.local_server_url
        return self.api_base

    def to_llm_config(self) -> "LLMConfig":
        from .llm_backend import LLMConfig
        return LLMConfig(
            model=self.model,
            api_base=self.api_base,
            api_key_env=self.api_key_env,
            temperature=self.temperature,
            timeout_s=self.timeout_s,
            use_local_server=self.use_local_server,
            local_server_url=self.local_server_url,
        )
