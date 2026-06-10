"""ItemSet: keyed, per-item artifact container (architecture A1).

Per-item artifacts (query texts, embeddings, retrieved results, per-item scores) used to
be aligned **positional lists** indexed by query position — brittle the moment a node
changes cardinality (augmentation N-variants, ``multi_query`` expansion) or drops a failed
item. ``ItemSet`` makes the identity explicit: an ordered list of ``ids`` with a ``values``
column aligned 1:1, so:

* **transform** keeps the ids (``with_values`` / ``map_values``),
* **failure / filter** drops ids → a *sparse* set; consumers ``align`` by id (gaps tolerated),
* **fan-out** mints lineage ids (``q42 → q42·aug0``) with the parent recoverable,
* **join** lines two sets up by id, not by position.

Columnar (ids + aligned values) rather than ``dict[id→value]`` so the vectorized batch path
is preserved: ``values`` may be a plain list *or* a 2-D ``np.ndarray`` (one row per id).

See ``evaluator-architecture.md`` §3.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Iterator, List, Sequence, Tuple

import numpy as np

# Separator between a parent id and a fan-out suffix (``q42·aug0``). Recoverable lineage.
LINEAGE_SEP = "·"


def child_id(parent: str, suffix: Any) -> str:
    """Mint a lineage child id, e.g. ``child_id("q42", "aug0") == "q42·aug0"``."""
    return f"{parent}{LINEAGE_SEP}{suffix}"


def parent_id(item_id: str) -> str:
    """Immediate parent of a (possibly nested) lineage id; itself if it has no parent."""
    return item_id.rsplit(LINEAGE_SEP, 1)[0] if LINEAGE_SEP in item_id else item_id


def root_id(item_id: str) -> str:
    """Original ancestor id (strips all lineage suffixes)."""
    return item_id.split(LINEAGE_SEP, 1)[0]


class ItemSet:
    """An ordered set of ``ids`` with a ``values`` column aligned 1:1.

    ``values`` is a list or a 2-D ``np.ndarray`` (row ``i`` belongs to ``ids[i]``). Ids must
    be unique. Instances are treated as immutable — every operation returns a new ``ItemSet``.
    """

    __slots__ = ("_ids", "_values")

    def __init__(self, ids: Sequence[str], values: Any) -> None:
        ids = [str(i) for i in ids]
        n_vals = values.shape[0] if isinstance(values, np.ndarray) else len(values)
        if len(ids) != n_vals:
            raise ValueError(
                f"ItemSet: {len(ids)} ids but {n_vals} values (must align 1:1)"
            )
        if len(set(ids)) != len(ids):
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"ItemSet: duplicate ids {dupes}")
        self._ids: List[str] = ids
        self._values: Any = values

    # ── constructors ──────────────────────────────────────────────────
    @classmethod
    def from_pairs(cls, pairs: Iterable[Tuple[str, Any]]) -> "ItemSet":
        """Build from ``(id, value)`` pairs (values become a list)."""
        ids: List[str] = []
        values: List[Any] = []
        for item_id, value in pairs:
            ids.append(str(item_id))
            values.append(value)
        return cls(ids, values)

    @classmethod
    def empty(cls) -> "ItemSet":
        return cls([], [])

    # ── access ────────────────────────────────────────────────────────
    @property
    def ids(self) -> List[str]:
        return list(self._ids)

    @property
    def values(self) -> Any:
        return self._values

    def __len__(self) -> int:
        return len(self._ids)

    def __iter__(self) -> Iterator[Tuple[str, Any]]:
        for i, item_id in enumerate(self._ids):
            yield item_id, self._row(i)

    def items(self) -> List[Tuple[str, Any]]:
        return list(iter(self))

    def has(self, item_id: str) -> bool:
        return item_id in self._index

    def value_for(self, item_id: str) -> Any:
        return self._row(self._index[item_id])

    def to_dict(self) -> dict:
        return {item_id: self._row(i) for i, item_id in enumerate(self._ids)}

    # ── transforms (ids preserved) ────────────────────────────────────
    def with_values(self, values: Any) -> "ItemSet":
        """Same ids, new aligned ``values`` (the output of a transform node)."""
        return ItemSet(self._ids, values)

    def map_values(self, fn: Callable[[Any], Any]) -> "ItemSet":
        """Elementwise map over values (list path); ids preserved."""
        return ItemSet(self._ids, [fn(self._row(i)) for i in range(len(self._ids))])

    # ── subset / reorder ──────────────────────────────────────────────
    def select(self, ids: Sequence[str]) -> "ItemSet":
        """Subset/reorder to exactly ``ids`` (every id must be present)."""
        missing = [i for i in ids if i not in self._index]
        if missing:
            raise KeyError(f"ItemSet.select: ids not present {missing}")
        rows = [self._index[i] for i in ids]
        return ItemSet(list(ids), self._take(rows))

    def filter(self, keep: Callable[[str, Any], bool]) -> "ItemSet":
        """Keep items where ``keep(id, value)`` is truthy (order preserved)."""
        rows = [i for i, item_id in enumerate(self._ids) if keep(item_id, self._row(i))]
        return ItemSet([self._ids[i] for i in rows], self._take(rows))

    def drop(self, ids: Iterable[str]) -> "ItemSet":
        """Drop the given ids (a failed-item / sparsity operation); order preserved."""
        drop_set = {str(i) for i in ids}
        rows = [i for i, item_id in enumerate(self._ids) if item_id not in drop_set]
        return ItemSet([self._ids[i] for i in rows], self._take(rows))

    def keep(self, ids: Iterable[str]) -> "ItemSet":
        """Keep only the given ids (order preserved by this set's order)."""
        keep_set = {str(i) for i in ids}
        rows = [i for i, item_id in enumerate(self._ids) if item_id in keep_set]
        return ItemSet([self._ids[i] for i in rows], self._take(rows))

    # ── fan-out (cardinality up, with lineage) ────────────────────────
    def fanout(self, fn: Callable[[str, Any], Iterable[Tuple[Any, Any]]]) -> "ItemSet":
        """Expand each item into children. ``fn(id, value) -> [(suffix, child_value), …]``;
        child ids are ``child_id(id, suffix)``. Values become a list."""
        new_ids: List[str] = []
        new_vals: List[Any] = []
        for i, item_id in enumerate(self._ids):
            for suffix, child_value in fn(item_id, self._row(i)):
                new_ids.append(child_id(item_id, suffix))
                new_vals.append(child_value)
        return ItemSet(new_ids, new_vals)

    # ── join (align two sets by id) ───────────────────────────────────
    def align(self, other: "ItemSet") -> Tuple[List[str], List[Any], List[Any]]:
        """Inner-join with ``other`` by id (this set's order). Returns
        ``(ids, self_values, other_values)`` over the shared ids — the basis for metric
        nodes (scored vs ground-truth) and cross-branch deltas."""
        ids: List[str] = []
        a: List[Any] = []
        b: List[Any] = []
        for i, item_id in enumerate(self._ids):
            if other.has(item_id):
                ids.append(item_id)
                a.append(self._row(i))
                b.append(other.value_for(item_id))
        return ids, a, b

    # ── internals ─────────────────────────────────────────────────────
    @property
    def _index(self) -> dict:
        # built lazily per call set; ids are small enough that this is fine
        return {item_id: i for i, item_id in enumerate(self._ids)}

    def _row(self, i: int) -> Any:
        return self._values[i]

    def _take(self, rows: List[int]) -> Any:
        if isinstance(self._values, np.ndarray):
            return self._values[rows] if rows else self._values[:0]
        return [self._values[i] for i in rows]

    def __repr__(self) -> str:
        kind = "ndarray" if isinstance(self._values, np.ndarray) else "list"
        return f"ItemSet(n={len(self._ids)}, values={kind})"
