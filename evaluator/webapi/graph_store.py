"""Persistence for named builder graphs — a directory of JSON files, one per saved graph.

Deliberately tiny (no DB): a saved graph is just the builder's canvas export
(``{mode, nodes, edges, branches?, llm?}``) written to ``<dir>/<name>.json``. The store
gives the builder a "save this graph / reopen a saved one" gallery without standing
infrastructure. Names are sanitised to a safe slug so a name can never escape the directory.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

_SAFE = re.compile(r"[^A-Za-z0-9 _-]+")


def _slug(name: str) -> str:
    """A filesystem-safe slug for a graph name (no separators → no traversal)."""
    slug = _SAFE.sub("", str(name)).strip()
    return slug[:80] if slug else ""


def default_graph_dir() -> Path:
    """Where graphs persist: ``$EVALUATOR_GRAPH_DIR`` else ``~/.evaluator/graphs``."""
    env = os.environ.get("EVALUATOR_GRAPH_DIR")
    return Path(env) if env else Path.home() / ".evaluator" / "graphs"


class GraphStore:
    """A directory of saved builder graphs (JSON per name)."""

    def __init__(self, directory: Optional[Path] = None) -> None:
        # The directory is created lazily on first save, so merely constructing a store (every
        # create_app) never touches the filesystem.
        self._dir = Path(directory) if directory is not None else default_graph_dir()

    def _path(self, name: str) -> Path:
        slug = _slug(name)
        if not slug:
            raise ValueError("Graph name must contain a letter, digit, space, '-' or '_'.")
        return self._dir / f"{slug}.json"

    def save(self, name: str, spec: Dict[str, Any]) -> str:
        """Write ``spec`` under ``name`` (overwriting an existing one). Returns the slug."""
        path = self._path(name)
        self._dir.mkdir(parents=True, exist_ok=True)
        payload = {"name": _slug(name), "spec": spec}
        path.write_text(json.dumps(payload, indent=2))
        return payload["name"]

    def list(self) -> List[Dict[str, Any]]:
        """Saved graphs as ``[{name, mode, n_nodes}]`` (a gallery index), sorted by name."""
        out: List[Dict[str, Any]] = []
        for path in sorted(self._dir.glob("*.json")):
            try:
                data = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                continue  # skip a corrupt/unreadable file rather than fail the listing
            spec = data.get("spec") or {}
            out.append({
                "name": data.get("name") or path.stem,
                "mode": spec.get("mode"),
                "n_nodes": len(spec.get("nodes") or []),
            })
        return out

    def get(self, name: str) -> Dict[str, Any]:
        """The saved canvas spec for ``name``. Raises ``KeyError`` if absent."""
        path = self._path(name)
        if not path.exists():
            raise KeyError(name)
        return json.loads(path.read_text()).get("spec") or {}

    def delete(self, name: str) -> None:
        """Remove a saved graph. Raises ``KeyError`` if absent."""
        path = self._path(name)
        if not path.exists():
            raise KeyError(name)
        path.unlink()
