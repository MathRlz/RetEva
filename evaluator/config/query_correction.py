"""Query-correction configuration (post-ASR domain correction).

The ``query_correction`` node rewrites the query text after ASR to repair domain errors
(e.g. drug/dose/unit confusions introduced by transcription) before embedding + retrieval —
the thesis "does post-ASR correction help Safety@k / CEER without losing Recall?" lever.

Mirrors the ``query_optimization`` node's place in the graph (a ``query_text → query_text``
transform), but is a *correction* step, configured separately. See
``evaluator-architecture.md`` §4/§5 and the Phase-3 plan (C1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from .llm_backend import LLMBackendMixin


@dataclass
class QueryCorrectionConfig(LLMBackendMixin):
    """Configuration for the post-ASR query-correction node.

    Attributes:
        enabled: Whether query correction runs. Default: False.
        method: Correction strategy. Default: "rule".
            Options: "rule" (deterministic word/phrase replacement). ``kb``/``llm`` are
            planned (knowledge-graph disambiguation / fine-tuned med-LLM rewrite).
        replacements: Extra ``wrong → right`` whole-word replacements (case-insensitive),
            merged over the default medical rules.
        use_default_rules: Include the built-in starter medical replacement rules.
    """

    enabled: bool = False
    method: str = "rule"  # rule | kb | llm
    replacements: Dict[str, str] = field(default_factory=dict)
    use_default_rules: bool = True
    # kb method: fuzzy-snap words to canonical medical terms (extends the built-in glossary).
    kb_terms: list = field(default_factory=list)
    kb_max_distance: int = 1
    # llm method: shared LLM backend (inherits EvaluationConfig.llm defaults via _merge_llm).
    model: str = "gpt-4o-mini"
    api_base: str = "https://api.openai.com/v1/chat/completions"
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.0
    timeout_s: int = 60
    use_local_server: bool = False
    local_server_url: Optional[str] = None

    def __post_init__(self) -> None:
        # The corrector registry is the authority (P3): a custom
        # @register_corrector method is valid config the moment it registers.
        from ..evaluation.query_correction import list_correctors

        valid = set(list_correctors())
        if self.method not in valid:
            raise ValueError(
                f"query_correction.method must be one of {sorted(valid)}, "
                f"got {self.method!r}"
            )
