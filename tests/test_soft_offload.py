"""Roadmap 2c: soft-CPU offload warm pool on ModelServiceProvider (LRU + TTL + reuse)."""

from evaluator.services.model_provider import ModelServiceProvider


class _FakeService:
    """Minimal FactoryModelService stand-in for the provider's soft-offload path."""

    def __init__(self, model, label):
        self._instance = model
        self.label = label
        self.moved_to = None
        self.stopped = False

    def move_to_device(self, device):
        self.moved_to = device

    def stop(self):
        self.stopped = True
        self._instance = None

    def get(self):
        return self._instance


class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def _provider(max_warm, ttl=None, clock=None):
    return ModelServiceProvider(
        soft_offload_max_warm=max_warm, soft_offload_ttl_s=ttl,
        clock=clock or (lambda: 0.0),
    )


def _put(p, key, model, device="cuda:0"):
    svc = _FakeService(model, label=f"asr:whisper@{device}")
    p._asr_services[key] = svc
    return svc


def test_soft_offload_parks_warm_keeps_service():
    p = _provider(max_warm=2)
    m = object()
    svc = _put(p, ("k1",), m)
    assert p.release_model_instance(m, soft_cpu=True) is True
    assert svc.moved_to == "cpu" and not svc.stopped     # parked, not freed
    assert p._asr_services.get(("k1",)) is svc           # still reachable by key
    assert p.offload_stats()["soft_offloads"] == 1


def test_full_free_is_the_default():
    p = _provider(max_warm=2)
    m = object()
    svc = _put(p, ("k1",), m)
    assert p.release_model_instance(m) is True            # soft_cpu not requested
    assert svc.stopped and ("k1",) not in p._asr_services
    assert p.offload_stats() == {
        "soft_offloads": 0, "evictions": 0, "warm_reuses": 0, "full_offloads": 1,
    }


def test_soft_cpu_falls_back_to_free_when_capacity_zero():
    p = _provider(max_warm=0)
    m = object()
    svc = _put(p, ("k1",), m)
    p.release_model_instance(m, soft_cpu=True)
    assert svc.stopped and p.offload_stats()["full_offloads"] == 1


def test_lru_eviction_over_capacity():
    p = _provider(max_warm=1)
    a, b = object(), object()
    sa = _put(p, ("a",), a)
    sb = _put(p, ("b",), b)
    p.release_model_instance(a, soft_cpu=True)            # a warm
    p.release_model_instance(b, soft_cpu=True)            # b warm → a is LRU-evicted
    assert sa.stopped and ("a",) not in p._asr_services   # a freed
    assert not sb.stopped and ("b",) in p._asr_services   # b stays warm
    stats = p.offload_stats()
    assert stats["soft_offloads"] == 2 and stats["evictions"] == 1


def test_ttl_eviction():
    clock = _Clock()
    p = _provider(max_warm=5, ttl=10.0, clock=clock)
    a, b = object(), object()
    sa = _put(p, ("a",), a)
    sb = _put(p, ("b",), b)
    p.release_model_instance(a, soft_cpu=True)            # parked at t=0
    clock.t = 20.0
    p.release_model_instance(b, soft_cpu=True)            # at t=20 → a is past TTL
    assert sa.stopped and not sb.stopped
    assert p.offload_stats()["evictions"] == 1


def test_warm_reuse_reactivates_to_device():
    p = _provider(max_warm=2)
    m = object()
    svc = _put(p, ("k1",), m, device="cuda:1")
    p.release_model_instance(m, soft_cpu=True)            # parked on cpu
    assert svc.moved_to == "cpu"
    # Re-request the same key → reactivated back to its device, unparked.
    got = p._get_or_create(p._asr_services, ("k1",), lambda: object(), "asr:whisper@cuda:1")
    assert got is m and svc.moved_to == "cuda:1"
    assert p.offload_stats()["warm_reuses"] == 1
    assert ("k1",) not in p._warm                         # no longer parked
