"""Roadmap 3b: off-RAM corpus/index — Parquet payload store + mmap FAISS store."""

import numpy as np
import pytest

from evaluator.storage.payload_store import ParquetPayloadStore
from evaluator.storage.vector_store import FaissMmapVectorStore, FaissVectorStore


# ── ParquetPayloadStore ──────────────────────────────────────────────


def test_payload_store_round_trips_by_id(tmp_path):
    payloads = [{"doc_id": f"d{i}", "text": f"document {i}"} for i in range(10)]
    store = ParquetPayloadStore.write(payloads, tmp_path / "p.parquet", row_group_size=4)
    assert len(store) == 10
    for i, p in enumerate(payloads):
        assert store.get(i) == p


def test_payload_store_out_of_range_is_none(tmp_path):
    store = ParquetPayloadStore.write([{"a": 1}], tmp_path / "p.parquet")
    assert store.get(-1) is None
    assert store.get(1) is None


def test_payload_store_get_many_preserves_input_order(tmp_path):
    payloads = [{"i": i} for i in range(12)]
    store = ParquetPayloadStore.write(payloads, tmp_path / "p.parquet", row_group_size=4)
    # ids spanning multiple row groups, scrambled — result must align 1:1 with the request
    assert store.get_many([7, 0, 11, 3]) == [{"i": 7}, {"i": 0}, {"i": 11}, {"i": 3}]


def test_payload_store_fetch_crosses_row_group_boundaries(tmp_path):
    payloads = [{"i": i} for i in range(9)]
    store = ParquetPayloadStore.write(payloads, tmp_path / "p.parquet", row_group_size=3)
    # row groups [0,1,2] [3,4,5] [6,7,8]; touch each
    assert [store.get(i)["i"] for i in (2, 5, 8, 0)] == [2, 5, 8, 0]


# ── FaissMmapVectorStore ─────────────────────────────────────────────


def _vectors(n=20, dim=8, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, dim)).astype("float32")


def test_mmap_store_search_matches_in_ram_faiss():
    # Off-RAM is a memory trade, not a different ranking: identical results to FaissVectorStore.
    vecs = _vectors()
    payloads = [{"doc_id": f"d{i}"} for i in range(len(vecs))]
    ram = FaissVectorStore(8)
    ram.build(vecs.copy(), list(payloads))
    mmap = FaissMmapVectorStore(8, row_group_size=4)
    mmap.build(vecs.copy(), list(payloads))

    queries = _vectors(n=6, seed=99)
    ram_res = ram.search_batch(queries.copy(), 5)
    mmap_res = mmap.search_batch(queries.copy(), 5)
    for r, m in zip(ram_res, mmap_res):
        assert [p["doc_id"] for p, _ in r] == [p["doc_id"] for p, _ in m]
        assert all(abs(rs - ms) < 1e-5 for (_, rs), (_, ms) in zip(r, m))


def test_mmap_store_keeps_payloads_off_ram():
    # The whole corpus must NOT be held in a RAM list (that is the point); it lives in the
    # Parquet store, fetched by id.
    mmap = FaissMmapVectorStore(8, row_group_size=4)
    mmap.build(_vectors(n=12), [{"doc_id": f"d{i}"} for i in range(12)])
    assert isinstance(mmap._store, ParquetPayloadStore)
    assert not getattr(mmap, "payloads", None)  # no all-in-RAM payload list
    assert mmap._payload_at(5) == {"doc_id": "d5"}
    assert mmap._payload_at(99) is None  # out of range, no raise


def test_mmap_store_save_load_round_trip(tmp_path):
    vecs = _vectors(n=12)
    payloads = [{"doc_id": f"d{i}"} for i in range(12)]
    mmap = FaissMmapVectorStore(8, row_group_size=4)
    mmap.build(vecs.copy(), list(payloads))
    before = mmap.search_batch(vecs[:3].copy(), 3)

    mmap.save(tmp_path / "store")
    reloaded = FaissMmapVectorStore(8, row_group_size=4)
    reloaded.load(tmp_path / "store")
    after = reloaded.search_batch(vecs[:3].copy(), 3)
    for b, a in zip(before, after):
        assert [p["doc_id"] for p, _ in b] == [p["doc_id"] for p, _ in a]


def test_factory_builds_faiss_mmap_store():
    from evaluator.config.vector_db import VectorDBConfig
    from evaluator.storage.registry import create_vector_store as create_vector_store_from_config

    store = create_vector_store_from_config(VectorDBConfig(type="faiss_mmap"), embedding_dim=8)
    assert isinstance(store, FaissMmapVectorStore)
    # faiss family still requires a dimension
    with pytest.raises(ValueError, match="embedding_dim"):
        create_vector_store_from_config(VectorDBConfig(type="faiss_mmap"), embedding_dim=None)
