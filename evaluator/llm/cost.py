"""LLM cost accounting: tokens, latency, and an optional budget cap (task T8).

A voice→IR comparison should report *compute cost per method*, not only accuracy — an answer
that needs a 30B judge per query is not "free". Every LLM call (judge / answer_gen /
query_optimization) records its token usage + wall time here, tagged by component, and the run
surfaces the totals in ``report.provenance.cost``. An optional token budget stops a run before it
runs away.

Process-scoped + thread-safe: one run executes at a time per process (the webapi runs each job as
a subprocess), but within a run T5 may fire LLM nodes from several branches concurrently, so the
accumulator is locked. ``reset()`` is called at run start.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, Optional


class BudgetExceededError(RuntimeError):
    """Raised when cumulative LLM token usage exceeds the configured budget (T8)."""


class CostTracker:
    """Thread-safe accumulator of per-component LLM token usage + latency."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_component: Dict[str, Dict[str, float]] = {}
        self._budget_tokens: Optional[int] = None

    def reset(self, *, budget_tokens: Optional[int] = None) -> None:
        with self._lock:
            self._by_component = {}
            self._budget_tokens = budget_tokens if budget_tokens else None

    def record(
        self,
        component: str,
        *,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        latency_s: float = 0.0,
    ) -> None:
        """Add one call's usage; raise ``BudgetExceededError`` if it pushes total over budget."""
        with self._lock:
            c = self._by_component.setdefault(
                component,
                {
                    "calls": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "latency_s": 0.0,
                },
            )
            c["calls"] += 1
            c["prompt_tokens"] += prompt_tokens
            c["completion_tokens"] += completion_tokens
            c["total_tokens"] += prompt_tokens + completion_tokens
            c["latency_s"] += latency_s
            total = sum(v["total_tokens"] for v in self._by_component.values())
            budget = self._budget_tokens
        if budget is not None and total > budget:
            raise BudgetExceededError(
                f"LLM token budget exceeded: {total} > {budget} tokens"
            )

    def summary(self) -> Optional[Dict[str, Any]]:
        """The ``report.provenance.cost`` block, or ``None`` if no LLM calls happened."""
        with self._lock:
            if not self._by_component:
                return None
            by_component = {
                k: {
                    kk: (round(vv, 3) if kk == "latency_s" else int(vv))
                    for kk, vv in v.items()
                }
                for k, v in self._by_component.items()
            }
            totals = {
                key: sum(v[key] for v in self._by_component.values())
                for key in (
                    "calls",
                    "prompt_tokens",
                    "completion_tokens",
                    "total_tokens",
                )
            }
            totals["latency_s"] = round(
                sum(v["latency_s"] for v in self._by_component.values()), 3
            )
            return {
                "by_component": by_component,
                "totals": {
                    k: (v if k == "latency_s" else int(v)) for k, v in totals.items()
                },
                "budget_tokens": self._budget_tokens,
            }


# Process-scoped singleton (mirrors the module-level LLM response cache in client.py).
COST = CostTracker()
