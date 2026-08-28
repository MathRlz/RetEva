"""Roadmap §8 — batch-size estimation (replaces OOM-and-halve with an estimate)."""

from types import SimpleNamespace

from evaluator.devices.memory import MemoryManager, suggest_batch_size


def _manager_with_snapshots(snapshots):
    """A MemoryManager whose ``get_memory_snapshot`` yields ``snapshots`` in order."""
    mgr = MemoryManager.__new__(MemoryManager)  # bypass real __init__ (no GPU needed)
    it = iter(snapshots)
    mgr.get_memory_snapshot = lambda device_idx=0: next(it)
    return mgr


def test_fits_to_free_memory_with_headroom():
    # 16 GB free, 0.5 GB/item, safety 1.5 → floor(16 / 0.75) = 21
    assert suggest_batch_size(16, 0.5) == 21


def test_clamped_to_max_and_min():
    assert suggest_batch_size(100, 0.01, max_batch=64) == 64
    assert suggest_batch_size(0.1, 0.5) == 1  # would be 0 → min_batch


def test_unknown_inputs_fall_back_to_max_batch():
    # non-positive free or per-item (unknown) → today's static max-batch behaviour
    assert suggest_batch_size(0, 0.5, max_batch=32) == 32
    assert suggest_batch_size(16, 0, max_batch=32) == 32


def test_safety_factor_lowers_the_estimate():
    high_safety = suggest_batch_size(16, 0.5, safety_factor=4.0)
    low_safety = suggest_batch_size(16, 0.5, safety_factor=1.0)
    assert high_safety < low_safety


# --- warm_up_batch_size (1b): one-sample warm-up → estimate from the used delta ---


def test_warm_up_runs_measure_once_and_estimates_from_delta():
    # used grows 1000 → 1500 MB ⇒ per_item ≈ 0.488 GB; 8 GB free, safety 1.5 ⇒ floor(10.9)=10
    mgr = _manager_with_snapshots([
        SimpleNamespace(used_mb=1000.0, free_mb=8192.0),
        SimpleNamespace(used_mb=1500.0, free_mb=8192.0),
    ])
    calls = []
    est = mgr.warm_up_batch_size(lambda: calls.append(1), max_batch=256)
    assert calls == [1]  # measure_fn ran exactly once
    assert est == 10


def test_warm_up_falls_back_to_max_batch_without_snapshots():
    # CPU / no CUDA → get_memory_snapshot returns None → today's static batch size
    mgr = _manager_with_snapshots([None])
    calls = []
    assert mgr.warm_up_batch_size(lambda: calls.append(1), max_batch=64) == 64
    assert calls == [1]  # still ran the warm-up forward, just didn't measure


def test_warm_up_falls_back_when_delta_is_non_positive():
    # No measurable growth (e.g. cached allocator) → fall back rather than divide by ~0
    mgr = _manager_with_snapshots([
        SimpleNamespace(used_mb=2000.0, free_mb=8192.0),
        SimpleNamespace(used_mb=2000.0, free_mb=8192.0),
    ])
    assert mgr.warm_up_batch_size(lambda: None, max_batch=128) == 128
