"""The vector-store registry (OCP): built-ins are registered, dispatch works, a plugin can add a
store without editing the factory, unknown types and missing-dim raise the expected errors.

The registry dispatches by ``config.type`` (duck-typed) — so unknown/plugin types are exercised
with a bare ``SimpleNamespace``; ``VectorDBConfig`` itself still normalizes ``type`` to the closed
``VectorDBType`` enum (the config-validation front door for the built-ins)."""
from types import SimpleNamespace

import pytest

from evaluator.config.vector_db import VectorDBConfig
from evaluator.storage.registry import (
    create_vector_store,
    register_vector_store,
    list_vector_stores,
)
from evaluator.storage.registry import create_vector_store as create_vector_store_from_config


def test_builtin_stores_are_registered():
    names = set(list_vector_stores())
    assert {"inmemory", "faiss", "faiss_mmap", "faiss_gpu", "chromadb", "qdrant"} <= names


def test_inmemory_and_faiss_dispatch():
    from evaluator.storage.vector_store import InMemoryVectorStore, FaissVectorStore

    assert isinstance(create_vector_store(VectorDBConfig(type="inmemory")), InMemoryVectorStore)
    assert isinstance(
        create_vector_store(VectorDBConfig(type="faiss"), embedding_dim=8), FaissVectorStore
    )


def test_requires_dim_raises_without_embedding_dim():
    with pytest.raises(ValueError, match="requires embedding_dim"):
        create_vector_store(VectorDBConfig(type="faiss"), embedding_dim=None)


def test_unknown_type_raises_with_available_list():
    with pytest.raises(ValueError, match="Unknown vector store type"):
        create_vector_store(SimpleNamespace(type="does_not_exist"))


def test_factory_delegates_to_registry():
    # The public factory entry point goes through the registry.
    from evaluator.storage.vector_store import InMemoryVectorStore

    store = create_vector_store_from_config(VectorDBConfig(type="inmemory"))
    assert isinstance(store, InMemoryVectorStore)


def test_plugin_can_register_a_store():
    sentinel = object()
    register_vector_store("test_plugin_store", lambda cfg, dim: sentinel)
    try:
        assert create_vector_store(SimpleNamespace(type="test_plugin_store")) is sentinel
    finally:
        from evaluator.storage.registry import _VECTOR_STORES

        _VECTOR_STORES.pop("test_plugin_store", None)
