"""On-disk payload store for off-RAM corpora (Roadmap 3b).

Backs the N corpus payloads with an Arrow/Parquet file so retrieval fetches docs by row id on
demand — one row group resident at a time — instead of holding the whole corpus in RAM. Pairs
with a memory-mapped vector index (``FaissMmapVectorStore``) so neither the index nor the
payloads are bounded by one box's RAM. Payloads are JSON-encoded per row, so an arbitrary
dict/str doc round-trips unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Optional, Sequence, Union


class ParquetPayloadStore:
    """Fetch-by-row-id payload store over a Parquet file (one row group resident at a time)."""

    def __init__(self, path: Union[str, Path], *, row_group_size: int = 1024) -> None:
        import pyarrow.parquet as pq

        self._path = str(path)
        self._pf = pq.ParquetFile(self._path)
        self._n = int(self._pf.metadata.num_rows)
        self._rg_size = int(row_group_size)
        self._cache_rg: Optional[int] = None      # the resident row group index
        self._cache_rows: Optional[List[str]] = None

    @classmethod
    def write(
        cls,
        payloads: Sequence[Any],
        path: Union[str, Path],
        *,
        row_group_size: int = 1024,
    ) -> "ParquetPayloadStore":
        """Write ``payloads`` to ``path`` (JSON-per-row) in fixed-size row groups, then open it."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        table = pa.table({"payload": [json.dumps(p, default=str) for p in payloads]})
        pq.write_table(table, str(path), row_group_size=row_group_size)
        return cls(path, row_group_size=row_group_size)

    def __len__(self) -> int:
        return self._n

    def get(self, idx: int) -> Any:
        """The payload at row ``idx`` (decoded), or ``None`` if out of range. Reads only the row
        group containing ``idx`` (cached), so memory is bounded by ``row_group_size`` rows."""
        if not 0 <= idx < self._n:
            return None
        rg = idx // self._rg_size
        if self._cache_rg != rg:
            col = self._pf.read_row_group(rg, columns=["payload"]).column("payload")
            self._cache_rows = col.to_pylist()
            self._cache_rg = rg
        return json.loads(self._cache_rows[idx - rg * self._rg_size])

    def get_many(self, indices: Sequence[int]) -> List[Any]:
        """Fetch a set of row ids (fetch-by-id-set), aligned 1:1 with ``indices``. Adjacent ids
        in the same row group reuse the cache, so a sorted id set reads each group once."""
        return [self.get(int(i)) for i in indices]
