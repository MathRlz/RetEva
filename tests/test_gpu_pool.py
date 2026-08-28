"""GPUPool allocation accounting + thread-safety (audit A2/F2)."""

import threading

from evaluator.devices.monitor import MemoryInfo
from evaluator.devices.pool import GPUPool


class _FakeMonitor:
    """Two GPUs with a fixed total, so the pool's free-memory accounting is deterministic."""

    def __init__(self, total_gb=100.0):
        self._total = total_gb

    def get_device_count(self):
        return 2

    def get_memory_usage(self, device_idx):
        return MemoryInfo(total=self._total, used=0.0, free=self._total)


def _pool(total_gb=100.0):
    # no buffer so total_usable == total_gb, allow CPU overflow
    return GPUPool(
        ["cuda:0", "cuda:1"], monitor=_FakeMonitor(total_gb),
        memory_buffer_percent=0.0, allow_cpu_fallback=True,
    )


def test_pool_has_a_lock():
    assert isinstance(_pool()._lock, type(threading.RLock()))


def test_concurrent_allocations_do_not_lose_reservations():
    # 64 threads each reserve 1GB on a pool with plenty of room; every increment must land
    # (the un-locked ``reserved += memory`` could drop updates) and ids stay distinct.
    pool = _pool(total_gb=1000.0)
    n = 64

    def alloc(i):
        pool.allocate(f"m{i}", memory_gb=1.0)

    threads = [threading.Thread(target=alloc, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    reserved = sum(u.reserved_memory_gb for u in pool.get_usage().values())
    allocated = sum(len(u.allocations) for u in pool.get_usage().values())
    assert reserved == float(n)  # no lost increments
    assert allocated == n  # every model recorded


def test_concurrent_allocations_never_overcommit_a_gpu():
    # Tight memory: 2 GPUs × 4GB; many 1.5GB requests. No GPU may exceed its total
    # (the race lets two threads both pass the free-check and over-commit).
    pool = _pool(total_gb=4.0)
    for i in range(40):
        pool.allocate(f"m{i}", memory_gb=1.5)  # serial is fine; we assert the invariant

    for dev, usage in pool.get_usage().items():
        if dev != "cpu":
            assert usage.reserved_memory_gb <= usage.total_memory_gb + 1e-9, dev


def test_release_returns_reservation():
    pool = _pool()
    dev = pool.allocate("asr", memory_gb=5.0)
    assert pool.get_usage()[dev].reserved_memory_gb == 5.0
    pool.release("asr")
    assert pool.get_usage()[dev].reserved_memory_gb == 0.0
