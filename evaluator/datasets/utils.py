"""Dataset loading utilities with format auto-detection.

Provides common utilities for loading JSON, JSONL, and CSV files,
with automatic format detection and schema validation.
"""

from __future__ import annotations

import csv
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


def load_json(path: Union[str, Path]) -> Union[Dict[str, Any], List[Any]]:
    """Load a JSON file into a dict or list.
    Raises FileNotFoundError if missing, json.JSONDecodeError on invalid JSON."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load a JSONL file (one JSON object per line), skipping blank lines.
    Raises FileNotFoundError if missing, ValueError on an invalid line."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")

    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{idx}: {exc}") from exc
    return rows


def detect_format(path: Union[str, Path]) -> str:
    """Detect a data file's format from its extension: "json", "jsonl", or "csv".
    Raises ValueError for unsupported extensions."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".json":
        return "json"
    elif suffix == ".jsonl":
        return "jsonl"
    elif suffix == ".csv":
        return "csv"
    else:
        raise ValueError(
            f"Unsupported file format: {suffix}\n"
            f"Supported formats: .json, .jsonl, .csv"
        )


def load_data_file(path: Union[str, Path]) -> Union[Dict[str, Any], List[Any]]:
    """Load a .json/.jsonl/.csv file with automatic format detection.
    Raises FileNotFoundError if missing, ValueError on unsupported format or invalid data."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    fmt = detect_format(path)

    if fmt == "json":
        return load_json(path)
    elif fmt == "jsonl":
        return load_jsonl(path)
    elif fmt == "csv":
        rows: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(dict(row))
        return rows
    else:
        raise ValueError(f"Unsupported format: {fmt}")


class DatasetLoader:
    """Dataset loader with format auto-detection and schema validation.

    Example:
        loader = DatasetLoader("data/questions.jsonl")
        data = loader.load()
        if loader.validate_schema(["question_id", "question_text"]):
            print("Schema is valid")
    """

    def __init__(self, path: Union[str, Path]):
        """Initialize the dataset loader."""
        self.path = Path(path)
        self._data: Optional[Union[Dict[str, Any], List[Any]]] = None
        self._format: Optional[str] = None

    def detect_format(self) -> str:
        """Detect and return the file format: "json", "jsonl", or "csv"."""
        if self._format is None:
            self._format = detect_format(self.path)
        return self._format

    def load(self) -> Union[Dict[str, Any], List[Any]]:
        """Load the dataset (cached after first call); returns parsed data (dict or list).
        Raises FileNotFoundError if missing, ValueError on unsupported/invalid data."""
        if self._data is None:
            self._data = load_data_file(self.path)
        return self._data

    def validate_schema(self, required_fields: List[str]) -> bool:
        """Return True if the dataset has all required fields: top-level keys
        for dict datasets, per-item fields for list datasets."""
        data = self.load()

        if isinstance(data, dict):
            return all(field in data for field in required_fields)

        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    return False
                if not all(field in item for field in required_fields):
                    return False
            return True

        return False


def normalize_text(text: str) -> str:
    """Normalize text for consistent comparison:
    Unicode NFC, lowercase, collapse whitespace runs, strip."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def extract_field(item: Any, field_path: str) -> Any:
    """Extract a field from a nested structure via dot notation ("metadata.author";
    numeric parts index lists: "items.0.id"). Supports dict, list/tuple, and
    attribute access; returns None if the path does not exist."""
    if item is None:
        return None

    parts = field_path.split(".")
    current = item

    for part in parts:
        if current is None:
            return None

        # Try dict access
        if isinstance(current, dict):
            current = current.get(part)
        # Try list/tuple access with numeric index
        elif isinstance(current, (list, tuple)):
            try:
                idx = int(part)
                if 0 <= idx < len(current):
                    current = current[idx]
                else:
                    return None
            except ValueError:
                return None
        # Try attribute access
        elif hasattr(current, part):
            current = getattr(current, part)
        else:
            return None

    return current
