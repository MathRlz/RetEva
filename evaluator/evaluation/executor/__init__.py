"""Executor subpackage: the DAG execution engine, run state, and entry points.

Split out of the former ``evaluation/phased.py`` god module (Phase 1, audit §1). The run
entry points (``run_graph`` / ``run_from_bundle``) and the run-state / context dataclasses
are re-exported here. Importing ``run`` also imports the handler package, so every stage
handler is registered as a side effect.
"""

from .state import RunState, RunFeatures, EvaluationContext
from .run import run_graph, run_from_bundle

# Importing the handler package fires every @register_stage_handler decorator, so the
# executor can dispatch by stage name. Done here so ``import executor`` is self-sufficient.
from .. import handlers as _handlers  # noqa: F401

__all__ = [
    "run_graph",
    "run_from_bundle",
    "RunState",
    "RunFeatures",
    "EvaluationContext",
]
