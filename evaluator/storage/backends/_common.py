"""Shared helpers for the optional ChromaDB and Qdrant vector store backends."""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def check_vector_payload_lengths(vectors, payloads) -> None:
    """Raise ValueError if the vector and payload counts don't match."""
    if len(vectors) != len(payloads):
        raise ValueError(
            f"Number of vectors ({len(vectors)}) must match "
            f"number of payloads ({len(payloads)})"
        )


def primitive_payload(payload: Any, idx: int, *, allow_lists: bool) -> Dict[str, Any]:
    """Build the metadata/payload dict a backend stores alongside one vector.

    Both backends need primitive-only values so the underlying client can
    serialize them (Qdrant additionally allows list values, ChromaDB does
    not), plus a ``_payload_idx`` field to recover the original payload on
    search. A non-dict payload is stored as its string form under
    ``_payload_str`` so retrieval still round-trips.
    """
    primitive_types = (str, int, float, bool, list) if allow_lists else (str, int, float, bool)
    if isinstance(payload, dict):
        metadata = {k: v for k, v in payload.items() if isinstance(v, primitive_types)}
        metadata["_payload_idx"] = idx
    else:
        metadata = {"_payload_idx": idx, "_payload_str": str(payload)}
    return metadata


def load_payloads_and_metadata(load_path: Path) -> Tuple[List[Any], Dict[str, Any]]:
    """Read the payloads.json/metadata.json pair written by a backend's ``save()``."""
    with open(load_path / "payloads.json", "r") as f:
        payloads = json.load(f)
    with open(load_path / "metadata.json", "r") as f:
        metadata = json.load(f)
    return payloads, metadata


def restore_common_metadata(store, metadata: Dict[str, Any]) -> None:
    """Set the ``collection_name``/``distance_fn`` attributes both backends restore
    from ``metadata.json`` on ``load()``; backend-specific fields stay in each ``load()``."""
    store.collection_name = metadata["collection_name"]
    store.distance_fn = metadata["distance_fn"]
