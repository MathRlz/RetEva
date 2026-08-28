"""InMemoryVectorStore: cosine build + nearest-neighbour search."""

import numpy as np

from evaluator.storage.vector_store import InMemoryVectorStore


def _store():
    store = InMemoryVectorStore()
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
    store.build(vectors, payloads=["east", "north", "west"])
    return store


def test_search_returns_nearest_payload_first():
    results = _store().search(np.array([0.9, 0.1], dtype=np.float32), k=3)
    payloads = [p for p, _ in results]
    assert payloads[0] == "east"
    assert payloads[-1] == "west"  # opposite direction ranks last


def test_search_is_magnitude_invariant():
    store = _store()
    small = store.search(np.array([0.1, 0.0], dtype=np.float32), k=1)
    big = store.search(np.array([100.0, 0.0], dtype=np.float32), k=1)
    assert small[0][0] == big[0][0] == "east"


def test_search_before_build_raises():
    import pytest

    with pytest.raises(ValueError, match="not built"):
        InMemoryVectorStore().search(np.array([1.0, 0.0], dtype=np.float32), k=1)


def test_search_batch_matches_per_query_search():
    store = _store()
    queries = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    batch = store.search_batch(queries, k=1)
    assert [b[0][0] for b in batch] == ["east", "north"]


def test_payload_at_is_bounds_checked():
    # A backend can return a stale/oversized _payload_idx; _payload_at returns None
    # (skipped hit) instead of IndexError or a wrong document (A4/F4).
    from evaluator.storage.vector_store import VectorStore

    class _S(VectorStore):
        def build(self, v, p):
            self._payloads = p

        def search(self, q, k):
            return []

        def save(self, p):
            pass

        def load(self, p):
            pass

    s = _S()
    s._payloads = ["a", "b", "c"]
    assert s._payload_at(0) == "a"
    assert s._payload_at(2) == "c"
    assert s._payload_at(3) is None  # out of range
    assert s._payload_at(-1) is None
    s._payloads = []
    assert s._payload_at(0) is None


def _capture(logger_name):
    # The project loggers set propagate=False once setup_logging runs (via another test), so
    # pytest's caplog (root handler) can't see them — attach a handler to the logger directly.
    import io
    import logging

    lg = logging.getLogger(logger_name)
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setLevel(logging.WARNING)
    lg.addHandler(handler)
    old = lg.level
    lg.setLevel(logging.WARNING)
    return lg, handler, buf, old


def test_search_drops_out_of_range_payloads():
    # H3: a corrupted/short payload list (e.g. a load() with mismatched sidecar) must drop the
    # bad hit + log, not IndexError mid-run.
    import numpy as np
    from evaluator.storage.vector_store import InMemoryVectorStore

    s = InMemoryVectorStore()
    s.build(np.eye(4, dtype="float32"), ["d0", "d1", "d2", "d3"])
    s.payloads = ["d0"]  # simulate stale/short payload sidecar
    lg, handler, buf, old = _capture("evaluator")
    try:
        hits = s.search(np.array([0, 0, 0, 1], dtype="float32"), k=4)
    finally:
        lg.removeHandler(handler)
        lg.setLevel(old)
    assert [p for p, _ in hits] == ["d0"]  # only the in-range payload survives
    assert "out of range" in buf.getvalue()


def test_search_batch_drops_out_of_range_payloads():
    import numpy as np
    from evaluator.storage.vector_store import InMemoryVectorStore

    s = InMemoryVectorStore()
    s.build(np.eye(3, dtype="float32"), ["a", "b", "c"])
    s.payloads = ["a", "b"]  # index 2 now out of range
    rows = s.search_batch(np.eye(3, dtype="float32"), k=3)
    flat = {p for row in rows for p, _ in row}
    assert flat == {"a", "b"}  # 'c' (index 2) dropped, no crash
