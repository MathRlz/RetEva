"""Roadmap 4b: order-preserving, determinism-neutral parallel map for CPU-bound stages."""

import pytest

from evaluator.config.evaluation import EvaluationConfig
from evaluator.evaluation.executor.cpu_parallel import (
    BACKENDS,
    parallel_map,
    resolve_cpu_backend,
)


# Module-level (picklable) functions so the "process" backend can serialize them.
def _square(x):
    return x * x


def _boom(x):
    if x == 3:
        raise ValueError("boom at 3")
    return x


@pytest.mark.parametrize("backend", BACKENDS)
def test_parallel_map_preserves_input_order(backend):
    items = list(range(20))
    assert parallel_map(_square, items, backend=backend, workers=4) == [i * i for i in items]


def test_all_backends_agree():
    items = list(range(50))
    ref = parallel_map(_square, items, backend="sync")
    assert parallel_map(_square, items, backend="thread", workers=4) == ref
    assert parallel_map(_square, items, backend="process", workers=4) == ref


def test_empty_input_returns_empty():
    for backend in BACKENDS:
        assert parallel_map(_square, [], backend=backend) == []


def test_invalid_backend_raises():
    with pytest.raises(ValueError, match="cpu_stage_executor"):
        parallel_map(_square, [1], backend="threads")  # typo


@pytest.mark.parametrize("backend", ["sync", "thread", "process"])
def test_worker_exception_propagates(backend):
    with pytest.raises(ValueError, match="boom"):
        parallel_map(_boom, [1, 2, 3, 4], backend=backend, workers=2)


def test_resolve_cpu_backend_defaults_and_validates():
    cfg = EvaluationConfig()
    assert resolve_cpu_backend(cfg) == ("sync", 0)  # inert default
    cfg.cpu_stage_executor = "process"
    cfg.cpu_stage_workers = 4
    assert resolve_cpu_backend(cfg) == ("process", 4)
    cfg.cpu_stage_executor = "bogus"
    with pytest.raises(ValueError, match="cpu_stage_executor"):
        resolve_cpu_backend(cfg)
