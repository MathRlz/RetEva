"""Answer generation configuration."""
from dataclasses import dataclass
from typing import Optional

from .llm_backend import LLMBackendMixin

# Where a retrieved doc's context text comes from.
CONTEXT_SOURCES = ("retrieved_text", "full_text")


@dataclass
class AnswerGenerationConfig(LLMBackendMixin):
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
        context_source: where each retrieved doc's context text comes from —
            ``retrieved_text`` (the indexed passage, default) or ``full_text`` (the article in
            the doc's ``metadata.full_text``, chunked and filtered to the chunks closest to the
            question). Docs with no full text fall back to their retrieved text.
        context_chunk_chars: chunk size when splitting a full article. Default: 1200.
        context_chunks: how many query-closest chunks per doc reach the prompt. Default: 4.

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
    # Sampling seed forwarded to the LLM request (inherited from the top-level
    # `llm:` block unless set here); None = omit. temperature 0 alone does not
    # pin an LLM's sampling.
    seed: Optional[int] = None
    compute_rouge: bool = True
    reference_metadata_field: str = "long_answer"
    context_source: str = "retrieved_text"  # retrieved_text | full_text
    context_chunk_chars: int = 1200
    context_chunks: int = 4

    def __post_init__(self) -> None:
        # Typo protection: a misspelled context_source would silently fall back to the indexed
        # passage, i.e. the full-text run would quietly measure the abstract run.
        if self.context_source not in CONTEXT_SOURCES:
            raise ValueError(
                f"answer_generation.context_source must be one of {list(CONTEXT_SOURCES)}, "
                f"got {self.context_source!r}"
            )
