"""Advanced (Tier-3) API namespace: extension-point exports.

This is the framework's *second* public surface. It is **orthogonal**, not
redundant, to the primary run-oriented surface:

- ``evaluator.public_api`` / ``evaluator.api`` (Tier-1) expose the high-level
  *execution* entry points re-exported at the top-level ``evaluator`` namespace:
  ``evaluate_from_config``, ``evaluate_from_preset``, ``quick_evaluate``,
  ``run_evaluation``, ``run_evaluation_matrix`` plus the core
  ``EvaluationConfig`` / ``EvaluationResults`` types. Use these to *run*
  evaluations.

- ``evaluator.advanced_api`` (Tier-3) exposes the building blocks you compose
  *inside* a custom pipeline or notebook: model classes, vector-store backends,
  dataset loaders/utilities, benchmark/timing helpers, and shared constants.
  Use these to *extend* or *introspect* the framework, not to drive a run.

Nothing here overlaps with the Tier-1 surface — there is no duplicated
``run_evaluation``-style helper. Each submodule is a thin, curated re-export of
deeper internal subpackages (``evaluator.models``, ``evaluator.storage``,
``evaluator.benchmarks``, ``evaluator.datasets``, ``evaluator.constants``), so
advanced users get one stable import path without reaching into private module
layout.

When to use which:
- "I want to run an evaluation"           -> ``from evaluator import ...``
- "I want a model/backend/loader/constant" -> ``from evaluator.advanced_api ...``

Import specific submodules:
- evaluator.advanced_api.models     - model implementation classes
- evaluator.advanced_api.storage    - vector-store backends
- evaluator.advanced_api.benchmarks - timing / benchmark helpers
- evaluator.advanced_api.datasets   - dataset loaders + parsing utilities
- evaluator.advanced_api.constants  - dimensions / defaults / thresholds
"""

from . import models, storage, benchmarks, datasets, constants

__all__ = ["models", "storage", "benchmarks", "datasets", "constants"]
