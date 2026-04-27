"""Judge configuration."""
from dataclasses import dataclass, field
from typing import Optional, Dict, List


@dataclass
class JudgeConfig:
    """
    Configuration for LLM-as-judge scoring.
    
    Optionally uses a language model to evaluate retrieval quality and relevance.
    Supports multi-aspect scoring, calibration, and local LLM models.
    
    Attributes:
        enabled: Whether LLM judge scoring is enabled. Default: False.
        api_base: Base URL for LLM API. Default: "https://api.openai.com/v1/chat/completions".
        model: Model identifier to use. Default: "gpt-4o-mini".
        api_key_env: Environment variable name for API key. Default: "OPENAI_API_KEY".
        temperature: Sampling temperature (0.0-1.0). Default: 0.0.
        max_cases: Maximum number of cases to evaluate. 0 = all available.
            Default: 50.
        timeout_s: Request timeout in seconds. Default: 60.
        
        judge_aspects: List of aspects to evaluate. Default: ["relevance"].
            Options: "relevance", "accuracy", "completeness", "clarity", "factuality".
        score_aggregation: How to combine aspect scores. Default: "average".
            Options: "average", "weighted", "min", "max".
        aspect_weights: Optional weights for each aspect (for weighted aggregation).
        output_format: Judge output format. Default: "score".
            Options: "score" (numeric only), "score_with_reasoning", "structured".
        few_shot_examples: Number of few-shot examples to include. Default: 0.
        chain_of_thought: Enable chain-of-thought prompting. Default: False.
        
        local_model_path: Path to local model (e.g., LLaMA, Mistral). Optional.
        batch_size: Batch size for local model inference. Default: 1.
    
    Examples:
        >>> # Basic single-aspect judging
        >>> config = JudgeConfig(
        ...     enabled=True,
        ...     model="gpt-4",
        ...     max_cases=100
        ... )
        >>> 
        >>> # Multi-aspect judging with reasoning
        >>> config = JudgeConfig(
        ...     enabled=True,
        ...     judge_aspects=["relevance", "accuracy", "completeness"],
        ...     score_aggregation="weighted",
        ...     aspect_weights={"relevance": 0.5, "accuracy": 0.3, "completeness": 0.2},
        ...     output_format="score_with_reasoning",
        ...     chain_of_thought=True
        ... )
    """
    enabled: bool = False
    api_base: str = "https://api.openai.com/v1/chat/completions"
    model: str = "gpt-4o-mini"
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.0
    max_cases: int = 50
    timeout_s: int = 60
    
    # Multi-aspect judging
    judge_aspects: List[str] = field(default_factory=lambda: ["relevance"])
    score_aggregation: str = "average"  # average | weighted | min | max
    aspect_weights: Optional[Dict[str, float]] = None
    
    # Output configuration
    output_format: str = "score"  # score | score_with_reasoning | structured
    few_shot_examples: int = 0
    chain_of_thought: bool = False
    
    # Local model support
    local_model_path: Optional[str] = None
    batch_size: int = 1
    
    # Local LLM server support
    use_local_server: bool = False
    local_server_url: Optional[str] = None

    # Judge mode and prompt overrides
    judge_mode: str = "retrieval"  # retrieval | answer_quality | both
    system_prompt: Optional[str] = None  # None → built-in default
    user_prompt_template: Optional[str] = None  # None → built-in default

    def __post_init__(self):
        """Validate configuration."""
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(f"temperature must be in [0, 2], got {self.temperature}")

        valid_judge_modes = {"retrieval", "answer_quality", "both"}
        if self.judge_mode not in valid_judge_modes:
            raise ValueError(
                f"judge_mode must be one of {valid_judge_modes}, got {self.judge_mode!r}"
            )
        
        valid_aspects = {"relevance", "accuracy", "completeness", "clarity", "factuality"}
        for aspect in self.judge_aspects:
            if aspect not in valid_aspects:
                raise ValueError(
                    f"Invalid judge aspect: '{aspect}'. Valid options: {valid_aspects}"
                )
        
        valid_aggregations = {"average", "weighted", "min", "max"}
        if self.score_aggregation not in valid_aggregations:
            raise ValueError(
                f"score_aggregation must be one of {valid_aggregations}, "
                f"got {self.score_aggregation}"
            )
        
        valid_formats = {"score", "score_with_reasoning", "structured"}
        if self.output_format not in valid_formats:
            raise ValueError(
                f"output_format must be one of {valid_formats}, "
                f"got {self.output_format}"
            )
        
        # Validate aspect weights if weighted aggregation
        if self.score_aggregation == "weighted":
            if not self.aspect_weights:
                raise ValueError(
                    "aspect_weights must be provided when score_aggregation='weighted'"
                )
            
            # Check all aspects have weights
            for aspect in self.judge_aspects:
                if aspect not in self.aspect_weights:
                    raise ValueError(f"Missing weight for aspect: {aspect}")
            
            # Validate weights sum to 1.0
            weight_sum = sum(self.aspect_weights.values())
            if not (0.99 <= weight_sum <= 1.01):  # Allow small floating point error
                raise ValueError(
                    f"aspect_weights must sum to 1.0, got {weight_sum}"
                )
    
    def get_api_base(self) -> str:
        """Get the appropriate API base URL (local or cloud)."""
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
