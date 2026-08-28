"""Load-time vector-store index/payload mismatch check."""

import json

import numpy as np
import pytest

from evaluator.storage.vector_store import InMemoryVectorStore


def test_vectorstore_load_rejects_count_mismatch(tmp_path):
    s = InMemoryVectorStore()
    s.build(np.eye(3, dtype="float32"), ["a", "b", "c"])
    s.save(str(tmp_path))
    # truncate the payload sidecar → load must fail loudly
    json.dump(["a"], (tmp_path / "payloads.json").open("w"))
    with pytest.raises(ValueError, match="index/payload mismatch"):
        InMemoryVectorStore().load(str(tmp_path))


def test_vectorstore_load_roundtrip_ok(tmp_path):
    s = InMemoryVectorStore()
    s.build(np.eye(2, dtype="float32"), ["x", "y"])
    s.save(str(tmp_path))
    loaded = InMemoryVectorStore()
    loaded.load(str(tmp_path))  # matching counts → no error
    assert loaded.payloads == ["x", "y"]
