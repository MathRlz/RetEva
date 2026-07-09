"""Deterministic per-item seeding (shared by handlers, augmenters, replay)."""

import hashlib


def item_seed(seed: int, query_id: str, node_id: str, variant: int = 0) -> int:
    """Deterministic 32-bit seed for one (item, node, variant) — order/parallelism-independent."""
    key = f"{seed}\x1f{query_id}\x1f{node_id}\x1f{variant}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")
