"""Shared utilities for evaluator WebAPI."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def artifact_listing(output_dir: Optional[str], *, max_entries: int = 200) -> List[Dict[str, Any]]:
    if not output_dir:
        return []
    base = Path(output_dir)
    if not base.exists() or not base.is_dir():
        return []

    files: List[Dict[str, Any]] = []
    for path in sorted(base.rglob("*")):
        if not path.is_file():
            continue
        rel_path = str(path.relative_to(base))
        stat = path.stat()
        files.append(
            {
                "path": rel_path,
                "size_bytes": stat.st_size,
                "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
            }
        )
        if len(files) >= max_entries:
            break
    return files
