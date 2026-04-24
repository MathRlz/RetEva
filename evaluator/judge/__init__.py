"""LLM-as-judge evaluation module.

This package provides LLM-based evaluation of retrieval quality and relevance.
Supports multi-aspect scoring, calibration, and local LLM models.

Main Components:
    - core: run_llm_judging and judge functions
    - prompts: Prompt templates for judge evaluation
    - calibration: Judge calibration utilities

Usage:
    from evaluator.judge import run_llm_judging
    
    judge_results = run_llm_judging(traces, config.judge)
"""

from .core import (
    run_llm_judging,
    judge_trace_with_openai_compatible,
    judge_multi_aspect,
    judge_single_aspect,
)

# Import prompts if they exist
try:
    from .prompts import (
        RELEVANCE_JUDGE_PROMPT,
        MULTI_ASPECT_JUDGE_PROMPT,
        FEW_SHOT_EXAMPLES,
    )
    _has_prompts = True
except ImportError:
    _has_prompts = False

# Import calibration if it exists
try:
    from .calibration import calibrate_judge, JudgeCalibrator
    _has_calibration = True
except ImportError:
    _has_calibration = False

__all__ = [
    "run_llm_judging",
    "judge_trace_with_openai_compatible",
    "judge_multi_aspect",
    "judge_single_aspect",
]

if _has_prompts:
    __all__.extend(["RELEVANCE_JUDGE_PROMPT", "MULTI_ASPECT_JUDGE_PROMPT", "FEW_SHOT_EXAMPLES"])

if _has_calibration:
    __all__.extend(["calibrate_judge", "JudgeCalibrator"])
