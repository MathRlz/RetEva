"""Query-correction configuration (post-ASR domain correction).

The ``query_correction`` node rewrites query text after ASR to repair domain errors
(e.g. drug/dose confusions) before embedding + retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from .llm_backend import LLMBackendMixin


@dataclass
class QueryCorrectionConfig(LLMBackendMixin):
    """Configuration for the post-ASR query-correction node."""

    enabled: bool = False
    method: str = "rule"  # rule | kb | phonetic | clinical | llm (registry is the authority)
    # Extra wrong→right whole-word replacements (case-insensitive), merged over the
    # default medical rules.
    replacements: Dict[str, str] = field(default_factory=dict)
    use_default_rules: bool = True
    # kb method: fuzzy-snap words to canonical medical terms (extends the built-in glossary).
    kb_terms: list = field(default_factory=list)
    kb_max_distance: int = 1
    # phonetic method: snap sound-alike drug names to the canonical vocabulary.
    drug_terms_path: Optional[str] = None  # None → bundled resources/drug_terms.json
    phonetic_max_edits: int = 2  # residual edit budget after a phonetic-code match
    # clinical method: dose/unit plausibility repair over the phonetic pass.
    drug_doses_path: Optional[str] = None  # None → bundled resources/drug_doses.json
    # llm method: snap out-of-vocab drug-shaped tokens in the LLM output back to
    # the drug vocabulary (post-hoc constrained decode — the model can't invent drugs).
    constrain_to_vocab: bool = False
    # Opt-in: publishes corrected-text WER/CER/CEER + the drug-aware ceer_rx metrics.
    # Default off — reports stay byte-identical unless set.
    corrected_metrics: bool = False
    # llm method: shared LLM backend (inherits EvaluationConfig.llm defaults via _merge_llm).
    model: str = "gpt-4o-mini"
    api_base: str = "https://api.openai.com/v1/chat/completions"
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.0
    timeout_s: int = 60
    use_local_server: bool = False
    local_server_url: Optional[str] = None
    # Sampling seed forwarded to the LLM request (inherited from top-level `llm:` unless
    # set); None = omit. temperature 0 alone does not pin sampling.
    seed: Optional[int] = None
    # Requests kept in flight for this component (inherited from top-level `llm:` unless set);
    # 1 = serial.
    concurrency: int = 1
