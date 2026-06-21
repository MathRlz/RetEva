"""LLM-as-judge evaluation module.

Multi-aspect LLM scoring of retrieval + answer quality. The judge runs over per-query traces and
produces per-query scores that flow through the normal metric path.

Usage:
    from evaluator.judge import run_llm_judging
    judge_results = run_llm_judging(traces, config.judge)
"""

from .core import run_llm_judging, judge_trace, build_judge_prompt

__all__ = ["run_llm_judging", "judge_trace", "build_judge_prompt"]
