"""Judge configuration."""
from dataclasses import dataclass, field
from typing import Optional, Dict, List

from .llm_backend import LLMBackendMixin

# Aspects the multi-aspect judge can score. retrieval-side: relevance; answer-side:
# faithfulness/correctness/completeness (+ legacy clarity/accuracy/factuality kept valid).
VALID_JUDGE_ASPECTS = {
    "relevance", "faithfulness", "correctness", "completeness",
    "clarity", "accuracy", "factuality",
}
JUDGE_TARGETS = {"retrieval", "answer_quality", "both"}
JUDGE_REFERENCE_MODES = {"free", "graded"}


@dataclass
class JudgeConfig(LLMBackendMixin):
    """LLM-as-judge scoring config.

    Off by default. When enabled, an LLM scores each query on the configured ``judge_aspects``
    (multi-aspect), producing per-query scores that flow through the normal metric path
    (``judge_overall`` / ``judge_<aspect>`` / ``judge_pass_rate``). The LLM endpoint/model
    inherit the global ``config.llm`` block unless overridden here.

    Attributes:
        enabled: master switch (default False).
        api_base/model/api_key_env/temperature/timeout_s: LLM endpoint (inherit ``config.llm``).
        use_local_server/local_server_url: route to a locally-spawned LLM server.
        judge_mode: what to evaluate — ``retrieval`` | ``answer_quality`` | ``both``.
        judge_aspects: dimensions to score, e.g. ``[relevance, faithfulness, completeness]``.
        score_aggregation: combine aspect scores → overall (``average`` | ``weighted`` | ``min``
            | ``max``); ``aspect_weights`` required for ``weighted``.
        reference_mode: ``free`` (intrinsic) or ``graded`` (feed the reference answer / relevant
            docs to the judge as a grading key).
        pass_threshold: overall score at/above which a case counts as PASS (judge_pass_rate).
        include_doc_text / judge_top_k: feed the top-k retrieved docs' *text* (not just ids).
        max_cases: cap on queries judged (0 = all).
        system_prompt / user_prompt_template: optional overrides (None → built-in).
    """
    enabled: bool = False
    api_base: str = "https://api.openai.com/v1/chat/completions"
    model: str = "gpt-4o-mini"
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.0
    max_cases: int = 50
    timeout_s: int = 60

    # What + how to judge
    judge_mode: str = "both"  # retrieval | answer_quality | both
    judge_aspects: List[str] = field(
        default_factory=lambda: ["relevance", "faithfulness", "completeness"]
    )
    score_aggregation: str = "average"  # average | weighted | min | max
    aspect_weights: Optional[Dict[str, float]] = None
    reference_mode: str = "free"  # free | graded
    pass_threshold: float = 0.5

    # Context fed to the judge
    include_doc_text: bool = True
    judge_top_k: int = 5

    # Local LLM server support
    use_local_server: bool = False
    local_server_url: Optional[str] = None

    # Prompt overrides
    system_prompt: Optional[str] = None  # None → built-in default
    user_prompt_template: Optional[str] = None  # None → built-in default

    def __post_init__(self):
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(f"temperature must be in [0, 2], got {self.temperature}")
        if self.judge_mode not in JUDGE_TARGETS:
            raise ValueError(f"judge_mode must be one of {JUDGE_TARGETS}, got {self.judge_mode!r}")
        if not self.judge_aspects:
            raise ValueError("judge_aspects must be a non-empty list")
        for aspect in self.judge_aspects:
            if aspect not in VALID_JUDGE_ASPECTS:
                raise ValueError(
                    f"Invalid judge aspect: '{aspect}'. Valid: {sorted(VALID_JUDGE_ASPECTS)}"
                )
        if self.score_aggregation not in {"average", "weighted", "min", "max"}:
            raise ValueError(f"score_aggregation invalid: {self.score_aggregation}")
        if self.reference_mode not in JUDGE_REFERENCE_MODES:
            raise ValueError(f"reference_mode must be one of {JUDGE_REFERENCE_MODES}")
        if not 0.0 <= self.pass_threshold <= 1.0:
            raise ValueError(f"pass_threshold must be in [0, 1], got {self.pass_threshold}")
        if self.score_aggregation == "weighted":
            if not self.aspect_weights:
                raise ValueError("aspect_weights required when score_aggregation='weighted'")
            for aspect in self.judge_aspects:
                if aspect not in self.aspect_weights:
                    raise ValueError(f"Missing weight for aspect: {aspect}")
            weight_sum = sum(self.aspect_weights.values())
            if not (0.99 <= weight_sum <= 1.01):
                raise ValueError(f"aspect_weights must sum to 1.0, got {weight_sum}")
