"""Shared helpers for the analysis exporters (B1/F14).

``_escape_latex`` and ``load_results`` previously had byte-divergent copies across
``export.py`` / ``significance.py`` / ``branch_report.py``. One canonical definition each
lives here; the others import from this module.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Union

# Order matters: backslash first so later replacements don't double-escape it.
_LATEX_REPLACEMENTS = (
    ("\\", r"\textbackslash{}"),
    ("&", r"\&"),
    ("%", r"\%"),
    ("$", r"\$"),
    ("#", r"\#"),
    ("_", r"\_"),
    ("{", r"\{"),
    ("}", r"\}"),
    ("~", r"\textasciitilde{}"),
    ("^", r"\textasciicircum{}"),
    ("@", "{@}"),
)


def _escape_latex(text: str) -> str:
    """Escape LaTeX special characters in ``text`` (backslash handled first)."""
    result = text
    for old, new in _LATEX_REPLACEMENTS:
        result = result.replace(old, new)
    return result


def load_results(path: Union[str, Path]) -> Dict[str, Any]:
    """Load evaluation results from a JSON file.

    Raises ``FileNotFoundError`` if the file is missing, ``json.JSONDecodeError`` if it
    is not valid JSON.
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
