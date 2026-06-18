"""Vector-store registry — the OCP extension point for storage backends.

Each backend registers a factory ``(vector_db_config, embedding_dim) -> VectorStore`` under its
type name; :func:`create_vector_store` dispatches by ``vector_db_config.type``. This mirrors the
model / node / handler / metric registries, so a plugin can add a store without editing the
pipeline factory. Optional-dependency backends (chromadb, qdrant) register an import guard
(package + install hint) that :func:`check_store_dependency` verifies at config-validation time,
so a missing package fails fast with a hint instead of mid-run.

The built-in stores are registered at import time below; the factories lazy-import their classes,
so importing this module never force-loads faiss / chromadb / qdrant.
"""

from typing import Any, Callable, Dict, NamedTuple, Optional


class _StoreSpec(NamedTuple):
    factory: Callable[[Any, Optional[int]], Any]
    requires_dim: bool
    optional_package: Optional[str]
    install_hint: Optional[str]


_VECTOR_STORES: Dict[str, _StoreSpec] = {}


def register_vector_store(
    name: str,
    factory: Callable[[Any, Optional[int]], Any],
    *,
    requires_dim: bool = False,
    optional_package: Optional[str] = None,
    install_hint: Optional[str] = None,
) -> None:
    """Register a vector-store factory under ``name``.

    ``requires_dim`` gates a ValueError when the embedding dimension is needed but absent.
    ``optional_package`` + ``install_hint`` declare an importlib spec checked by
    :func:`check_store_dependency` (for backends behind an extra)."""
    _VECTOR_STORES[name] = _StoreSpec(
        factory, requires_dim, optional_package, install_hint
    )


def list_vector_stores() -> list:
    """The registered store type names (sorted)."""
    return sorted(_VECTOR_STORES)


def _store_type(vector_db_config) -> str:
    t = vector_db_config.type
    return t.value if hasattr(t, "value") else str(t)


def check_store_dependency(vector_db_config) -> None:
    """Raise ``ImportError`` (with an install hint) when the configured store's optional package
    is missing. Call at validation time for immediate feedback instead of a mid-run crash."""
    import importlib

    spec = _VECTOR_STORES.get(_store_type(vector_db_config))
    if (
        spec is not None
        and spec.optional_package is not None
        and importlib.util.find_spec(spec.optional_package) is None
    ):
        raise ImportError(spec.install_hint)


def create_vector_store(vector_db_config, embedding_dim: Optional[int] = None):
    """Build the configured vector store via its registered factory."""
    store_type = _store_type(vector_db_config)
    spec = _VECTOR_STORES.get(store_type)
    if spec is None:
        available = ", ".join(list_vector_stores())
        raise ValueError(
            f"Unknown vector store type: '{store_type}'.\nAvailable types: {available}"
        )
    if spec.requires_dim and embedding_dim is None:
        raise ValueError(
            f"vector_db.type='{store_type}' requires embedding_dim. "
            "Set model.audio_emb_dim or model.text_emb_model_type so the dimension can be resolved."
        )
    return spec.factory(vector_db_config, embedding_dim)


# ── Built-in store factories (lazy-import their classes) ──────────────────────────────


def _make_inmemory(cfg, dim):
    from .vector_store import InMemoryVectorStore

    return InMemoryVectorStore()


def _make_faiss(cfg, dim):
    from .vector_store import FaissVectorStore

    return FaissVectorStore(dim)


def _make_faiss_mmap(cfg, dim):
    # Off-RAM corpus/index (3b): mmap index + Parquet payloads.
    from .vector_store import FaissMmapVectorStore

    return FaissMmapVectorStore(dim)


def _make_faiss_gpu(cfg, dim):
    from .vector_store import FaissGpuVectorStore

    gpu = getattr(cfg, "gpu_id", 0)
    return FaissGpuVectorStore(dim, gpu_id=gpu if isinstance(gpu, int) else 0)


def _make_chromadb(cfg, dim):
    try:
        from .backends.chromadb_store import ChromaDBVectorStore
    except ImportError:
        raise ImportError(
            "ChromaDB is not installed. Install it with: pip install chromadb"
        )
    return ChromaDBVectorStore(
        collection_name=cfg.chromadb_collection_name,
        persist_path=cfg.chromadb_path,
        distance_fn=cfg.distance_metric,
    )


def _make_qdrant(cfg, dim):
    try:
        from .backends.qdrant_store import QdrantVectorStore
    except ImportError:
        raise ImportError(
            "Qdrant client is not installed. Install it with: pip install qdrant-client"
        )
    return QdrantVectorStore(
        collection_name=cfg.qdrant_collection_name,
        url=cfg.qdrant_url,
        path=cfg.qdrant_path,
        distance_fn=cfg.distance_metric,
        api_key=cfg.qdrant_api_key,
    )


register_vector_store("inmemory", _make_inmemory)
register_vector_store("faiss", _make_faiss, requires_dim=True)
register_vector_store("faiss_mmap", _make_faiss_mmap, requires_dim=True)
register_vector_store("faiss_gpu", _make_faiss_gpu, requires_dim=True)
register_vector_store(
    "chromadb", _make_chromadb,
    optional_package="chromadb",
    install_hint="ChromaDB is not installed. Install with: pip install evaluator[chromadb]",
)
register_vector_store(
    "qdrant", _make_qdrant,
    optional_package="qdrant_client",
    install_hint="Qdrant client is not installed. Install with: pip install evaluator[qdrant]",
)
