"""Answer generation configuration."""
from dataclasses import dataclass
from typing import Optional

from .llm_backend import LLMBackendMixin

# Where a retrieved doc's context text comes from.
CONTEXT_SOURCES = ("retrieved_text", "full_text")


@dataclass
class AnswerGenerationConfig(LLMBackendMixin):
    """Configuration for RAG answer generation."""
    enabled: bool = False
    method: str = "simple"  # simple | chain_of_thought | multi_query
    system_prompt: Optional[str] = None  # None = built-in default
    # {question}/{context} placeholders; None = built-in default
    prompt_template: Optional[str] = None
    context_docs: int = 3
    context_max_chars: int = 600
    model: str = "gpt-4o-mini"
    api_base: str = "https://api.openai.com/v1/chat/completions"
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.0
    max_cases: int = 0  # 0 = all
    timeout_s: int = 120
    use_local_server: bool = False
    local_server_url: Optional[str] = None
    # Sampling seed forwarded to the LLM request (inherited from top-level `llm:` unless
    # set); None = omit. temperature 0 alone does not pin sampling.
    seed: Optional[int] = None
    # Requests kept in flight for this component (inherited from top-level `llm:` unless set);
    # 1 = serial.
    concurrency: int = 1
    compute_rouge: bool = True
    reference_metadata_field: str = "long_answer"
    # retrieved_text (indexed passage) | full_text (chunked article, closest chunks to the
    # question); falls back to retrieved_text when a doc has no full text.
    context_source: str = "retrieved_text"
    context_chunk_chars: int = 1200
    context_chunks: int = 4

    def __post_init__(self) -> None:
        if self.concurrency < 1:
            raise ValueError(
                f"answer_generation.concurrency must be >= 1, got {self.concurrency}"
            )
        # Typo protection: a misspelled context_source would silently fall back to the indexed
        # passage, i.e. the full-text run would quietly measure the abstract run.
        if self.context_source not in CONTEXT_SOURCES:
            raise ValueError(
                f"answer_generation.context_source must be one of {list(CONTEXT_SOURCES)}, "
                f"got {self.context_source!r}"
            )
