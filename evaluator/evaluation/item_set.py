"""ItemSet: keyed, per-item artifact container.

Replaces positional lists (aligned by query position — brittle once a node changes cardinality,
e.g. augmentation fan-out, or drops a failed item) with an ordered ``ids`` list and an aligned
``values`` column, so identity, not position, is the invariant transforms/filters/joins rely on.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

#: Separator between a lineage parent and its variant suffix (``q42·aug0``).
LINEAGE_SEPARATOR = "·"


def lineage_parent(item_id: str) -> str:
    """``q42·aug0`` → ``q42`` (identity for a non-variant id)."""
    return item_id.split(LINEAGE_SEPARATOR, 1)[0]


class ItemSet:
    """An ordered set of ``ids`` with a ``values`` column aligned 1:1.

    ``values`` is a list or a 2-D ``np.ndarray`` (row ``i`` belongs to ``ids[i]``). Ids must
    be unique. Instances are treated as immutable — every operation returns a new ``ItemSet``.
    """

    __slots__ = ("_ids", "_values", "_index_cache")

    def __init__(self, ids: Sequence[str], values: Any) -> None:
        ids = [str(i) for i in ids]
        n_vals = values.shape[0] if isinstance(values, np.ndarray) else len(values)
        if len(ids) != n_vals:
            raise ValueError(
                f"ItemSet: {len(ids)} ids but {n_vals} values (must align 1:1)"
            )
        if len(set(ids)) != len(ids):
            from collections import Counter

            dupes = sorted(i for i, c in Counter(ids).items() if c > 1)
            raise ValueError(f"ItemSet: duplicate ids {dupes}")
        self._ids: List[str] = ids
        self._values: Any = values
        # Lazily-built id→row index, cached because ItemSet is immutable.
        self._index_cache: dict = None

    # constructors
    @classmethod
    def empty(cls) -> "ItemSet":
        return cls([], [])

    @classmethod
    def concat(cls, parts: "Iterable[ItemSet]") -> "ItemSet":
        """Concatenate ItemSets in order into one (the windowed-execution accumulator). Ids
        stay in part-then-row order and must be globally unique (the ctor enforces it, so
        overlapping windows are a loud error). All-list parts stay a list; all-ndarray parts
        with matching width stack into one 2-D array. An empty input is the empty set."""
        parts = [p for p in parts]
        if not parts:
            return cls.empty()
        ids: List[str] = []
        for p in parts:
            ids.extend(p._ids)
        arrays = [p._values for p in parts if len(p) > 0]
        if arrays and all(isinstance(v, np.ndarray) for v in arrays):
            return cls(ids, np.concatenate(arrays, axis=0))
        values: List[Any] = []
        for p in parts:
            values.extend(list(p._values))
        return cls(ids, values)

    # access
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

    # transforms (ids preserved)
    def with_values(self, values: Any) -> "ItemSet":
        """Same ids, new aligned ``values`` (the output of a transform node)."""
        return ItemSet(self._ids, values)

    def map_values(self, fn: Callable[[Any], Any]) -> "ItemSet":
        """Elementwise map over values (list path); ids preserved."""
        return ItemSet(self._ids, [fn(self._row(i)) for i in range(len(self._ids))])

    # subset / reorder
    def filter(self, keep: Callable[[str, Any], bool]) -> "ItemSet":
        """Keep items where ``keep(id, value)`` is truthy (order preserved)."""
        rows = [i for i, item_id in enumerate(self._ids) if keep(item_id, self._row(i))]
        return ItemSet([self._ids[i] for i in rows], self._take(rows))

    # join (align two sets by id)
    def align(
        self,
        other: "ItemSet",
        *,
        other_key: Optional[Callable[[str], str]] = None,
    ) -> Tuple[List[str], List[Any], List[Any]]:
        """Inner-join with ``other`` by id (this set's order) -> ``(ids, self_values,
        other_values)`` over the shared ids — the basis for metric nodes (scored vs
        ground-truth) and cross-branch deltas. ``other_key`` maps an id that missed
        directly to the id to read from ``other`` instead (the lineage join: variant
        ``q42·aug0`` reads ``other``'s ``q42``). The returned ids stay this set's ids, so
        variants remain distinct items."""
        ids: List[str] = []
        a: List[Any] = []
        b: List[Any] = []
        for i, item_id in enumerate(self._ids):
            key = item_id
            if not other.has(key) and other_key is not None:
                key = other_key(item_id)
            if other.has(key):
                ids.append(item_id)
                a.append(self._row(i))
                b.append(other.value_for(key))
        return ids, a, b

    # internals
    @property
    def _index(self) -> dict:
        # Built once and cached — has/value_for/align run in loops, and rebuilding this dict
        # each time would make an align loop O(n²) instead of O(n). Safe because ItemSet is
        # immutable.
        idx = self._index_cache
        if idx is None:
            idx = {item_id: i for i, item_id in enumerate(self._ids)}
            self._index_cache = idx
        return idx

    def _row(self, i: int) -> Any:
        return self._values[i]

    def _take(self, rows: List[int]) -> Any:
        if isinstance(self._values, np.ndarray):
            return self._values[rows] if rows else self._values[:0]
        return [self._values[i] for i in rows]

    def __repr__(self) -> str:
        kind = "ndarray" if isinstance(self._values, np.ndarray) else "list"
        return f"ItemSet(n={len(self._ids)}, values={kind})"
