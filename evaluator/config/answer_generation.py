"""Answer generation configuration."""
from dataclasses import dataclass
from typing import Optional


@dataclass
class AnswerGenerationConfig:
    """Configuration for RAG answer generation (pipeline Phase 4.5).

    Attributes:
        enabled: Whether answer generation is enabled. Default: False.
        method: Generation method. Default: "simple".
            Options: "simple", "chain_of_thought", "multi_query".
        system_prompt: Custom system prompt. None → built-in default.
        prompt_template: Custom prompt template with {question} and {context} placeholders.
            None → built-in default.
        context_docs: Number of retrieved docs to include as context. Default: 3.
        context_max_chars: Max characters per context chunk. Default: 600.
        model: LLM model identifier. Default: "gpt-4o-mini".
        api_base: OpenAI-compatible endpoint URL.
        api_key_env: Env var name for API key. Default: "OPENAI_API_KEY".
        temperature: Sampling temperature. Default: 0.0.
        max_cases: Max queries to generate answers for. 0 = all. Default: 0.
        timeout_s: Request timeout in seconds. Default: 120.
        use_local_server: Use local_server_url instead of api_base. Default: False.
        local_server_url: URL for local LLM server (e.g. Ollama). Default: None.
        compute_rouge: Compute ROUGE-1/2/L vs reference answer. Default: True.
        reference_metadata_field: Corpus doc metadata field holding the reference answer.
            Default: "long_answer".

    Examples:
        >>> cfg = AnswerGenerationConfig(
        ...     enabled=True,
        ...     method="chain_of_thought",
        ...     use_local_server=True,
        ...     local_server_url="http://localhost:11434/v1/chat/completions",
        ...     model="llama3.2",
        ...     api_key_env="OLLAMA_API_KEY",
        ... )
    """
    enabled: bool = False
    method: str = "simple"  # simple | chain_of_thought | multi_query
    system_prompt: Optional[str] = None
    prompt_template: Optional[str] = None
    context_docs: int = 3
    context_max_chars: int = 600
    model: str = "gpt-4o-mini"
    api_base: str = "https://api.openai.com/v1/chat/completions"
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.0
    max_cases: int = 0
    timeout_s: int = 120
    use_local_server: bool = False
    local_server_url: Optional[str] = None
    compute_rouge: bool = True
    reference_metadata_field: str = "long_answer"

    def get_api_base(self) -> str:
        """Return effective API base URL (local server takes precedence)."""
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
