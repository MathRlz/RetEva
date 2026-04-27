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
    """Load a JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed JSON content (dict or list).

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file contains invalid JSON.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load a JSONL file (one JSON object per line).

    Args:
        path: Path to the JSONL file.

    Returns:
        List of parsed JSON objects.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If a line contains invalid JSON.
    """
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
    """Detect the format of a data file based on extension.

    Args:
        path: Path to the data file.

    Returns:
        Format string: "json", "jsonl", or "csv".

    Raises:
        ValueError: If the format is not supported.
    """
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
    """Load a data file with automatic format detection.

    Args:
        path: Path to the data file (.json, .jsonl, or .csv).

    Returns:
        Parsed data (dict or list).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the format is not supported or data is invalid.
    """
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
        """Initialize the dataset loader.

        Args:
            path: Path to the dataset file.
        """
        self.path = Path(path)
        self._data: Optional[Union[Dict[str, Any], List[Any]]] = None
        self._format: Optional[str] = None

    def detect_format(self) -> str:
        """Detect and return the file format.

        Returns:
            Format string: "json", "jsonl", or "csv".
        """
        if self._format is None:
            self._format = detect_format(self.path)
        return self._format

    def load(self) -> Union[Dict[str, Any], List[Any]]:
        """Load the dataset.

        Returns:
            Parsed data (dict or list).

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the format is not supported or data is invalid.
        """
        if self._data is None:
            self._data = load_data_file(self.path)
        return self._data

    def validate_schema(self, required_fields: List[str]) -> bool:
        """Validate that all items in the dataset have the required fields.

        Args:
            required_fields: List of field names that must be present.

        Returns:
            True if all items have all required fields, False otherwise.

        Note:
            For dict datasets, checks if required_fields are top-level keys.
            For list datasets, checks if each item has all required fields.
        """
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
    """Normalize text for consistent comparison.

    Performs the following normalizations:
    - Unicode normalization (NFC)
    - Lowercase conversion
    - Whitespace normalization (multiple spaces to single space)
    - Strip leading/trailing whitespace

    Args:
        text: Input text to normalize.

    Returns:
        Normalized text string.
    """
    if not text:
        return ""
    # Unicode normalization
    text = unicodedata.normalize("NFC", text)
    # Lowercase
    text = text.lower()
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    # Strip
    text = text.strip()
    return text


def extract_field(item: Any, field_path: str) -> Any:
    """Extract a field from a nested data structure.

    Supports dot notation for nested access (e.g., "metadata.author").

    Args:
        item: The data structure to extract from (dict, list, or object).
        field_path: Dot-separated path to the field (e.g., "user.name").

    Returns:
        The extracted value, or None if the path does not exist.

    Examples:
        >>> extract_field({"user": {"name": "Alice"}}, "user.name")
        'Alice'
        >>> extract_field({"items": [{"id": 1}]}, "items.0.id")
        1
    """
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
