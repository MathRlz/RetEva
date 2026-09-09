"""The flat metrics mapping returned by ``run_graph``: a plain dict of well-known keys
(meta/ASR/embedding/IR/diagnostics/feature groups, each present only when relevant). The
internal counterpart of the public :class:`~evaluator.evaluation.results.EvaluationResults`."""

from typing import Any, Dict

RunResults = Dict[str, Any]
