"""Query optimization configuration."""
from dataclasses import dataclass
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
        llm_api_base: Base URL for LLM API. Default: "https://api.openai.com/v1/chat/completions".
        llm_model: Model identifier for query optimization. Default: "gpt-4o-mini".
        llm_api_key_env: Environment variable name for API key. Default: "OPENAI_API_KEY".
        llm_temperature: Sampling temperature (0.0-1.0). Default: 0.3.
        max_iterations: Maximum refinement iterations for rewrite. Default: 2.
        use_initial_context: Use retrieval context for iterative refinement. Default: True.
        context_top_k: Number of top results to use as context. Default: 3.
        combine_strategy: How to combine multi-query results. Default: "rrf".
            Options: "rrf", "weighted", "union", "intersection".
        timeout_s: Request timeout in seconds. Default: 30.
    
    Examples:
        >>> # Iterative query rewriting
        >>> config = QueryOptimizationConfig(
        ...     enabled=True,
        ...     method="rewrite",
        ...     max_iterations=3
        ... )
        >>> 
        >>> # HyDE (Hypothetical Document Embeddings)
        >>> config = QueryOptimizationConfig(
        ...     enabled=True,
        ...     method="hyde",
        ...     llm_temperature=0.7
        ... )
        >>> 
        >>> # Query decomposition
        >>> config = QueryOptimizationConfig(
        ...     enabled=True,
        ...     method="decompose",
        ...     combine_strategy="weighted"
        ... )
    """
    enabled: bool = False
    method: str = "rewrite"  # rewrite | hyde | decompose | multi_query
    llm_api_base: str = "https://api.openai.com/v1/chat/completions"
    llm_model: str = "gpt-4o-mini"
    llm_api_key_env: str = "OPENAI_API_KEY"
    llm_temperature: float = 0.3
    max_iterations: int = 2
    use_initial_context: bool = True
    context_top_k: int = 3
    combine_strategy: str = "rrf"  # rrf | weighted | union | intersection
    timeout_s: int = 30
    
    # Local LLM server support
    use_local_server: bool = False
    local_server_url: Optional[str] = None
    
    def __post_init__(self):
        """Validate configuration."""
        valid_methods = {"rewrite", "hyde", "decompose", "multi_query"}
        if self.method not in valid_methods:
            raise ValueError(
                f"method must be one of {valid_methods}, got {self.method}"
            )
        
        valid_strategies = {"rrf", "weighted", "union", "intersection"}
        if self.combine_strategy not in valid_strategies:
            raise ValueError(
                f"combine_strategy must be one of {valid_strategies}, got {self.combine_strategy}"
            )
        
        if not 0.0 <= self.llm_temperature <= 2.0:
            raise ValueError(f"llm_temperature must be in [0, 2], got {self.llm_temperature}")
        
        if self.max_iterations < 1:
            raise ValueError(f"max_iterations must be >= 1, got {self.max_iterations}")
    
    def get_api_base(self) -> str:
        """Get the appropriate API base URL (local or cloud)."""
        if self.use_local_server and self.local_server_url:
            return self.local_server_url
        return self.llm_api_base
