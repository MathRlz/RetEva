"""Tests for dataset_utils module."""

import json
import csv
import tempfile
from pathlib import Path

import pytest

from evaluator.datasets.utils import (
    load_json,
    load_jsonl,
    detect_format,
    load_data_file,
    DatasetLoader,
    normalize_text,
    extract_field,
)


class TestLoadJson:
    """Tests for load_json function."""

    def test_load_json_dict(self, tmp_path):
        """Test loading a JSON file containing a dict."""
        data = {"key": "value", "number": 42}
        file_path = tmp_path / "test.json"
        file_path.write_text(json.dumps(data))

        result = load_json(file_path)
        assert result == data

    def test_load_json_list(self, tmp_path):
        """Test loading a JSON file containing a list."""
        data = [{"id": 1}, {"id": 2}]
        file_path = tmp_path / "test.json"
        file_path.write_text(json.dumps(data))

        result = load_json(file_path)
        assert result == data

    def test_load_json_file_not_found(self, tmp_path):
        """Test that FileNotFoundError is raised for missing files."""
        with pytest.raises(FileNotFoundError):
            load_json(tmp_path / "nonexistent.json")

    def test_load_json_invalid_json(self, tmp_path):
        """Test that JSONDecodeError is raised for invalid JSON."""
        file_path = tmp_path / "invalid.json"
        file_path.write_text("not valid json {")

        with pytest.raises(json.JSONDecodeError):
            load_json(file_path)


class TestLoadJsonl:
    """Tests for load_jsonl function."""

    def test_load_jsonl_basic(self, tmp_path):
        """Test loading a basic JSONL file."""
        file_path = tmp_path / "test.jsonl"
        lines = [
            json.dumps({"id": 1, "text": "first"}),
            json.dumps({"id": 2, "text": "second"}),
        ]
        file_path.write_text("\n".join(lines))

        result = load_jsonl(file_path)
        assert len(result) == 2
        assert result[0] == {"id": 1, "text": "first"}
        assert result[1] == {"id": 2, "text": "second"}

    def test_load_jsonl_empty_lines(self, tmp_path):
        """Test that empty lines are skipped."""
        file_path = tmp_path / "test.jsonl"
        content = '{"id": 1}\n\n{"id": 2}\n\n'
        file_path.write_text(content)

        result = load_jsonl(file_path)
        assert len(result) == 2

    def test_load_jsonl_file_not_found(self, tmp_path):
        """Test that FileNotFoundError is raised for missing files."""
        with pytest.raises(FileNotFoundError):
            load_jsonl(tmp_path / "nonexistent.jsonl")

    def test_load_jsonl_invalid_line(self, tmp_path):
        """Test that ValueError is raised for invalid JSON lines."""
        file_path = tmp_path / "invalid.jsonl"
        file_path.write_text('{"valid": true}\nnot valid json\n')

        with pytest.raises(ValueError, match="Invalid JSONL"):
            load_jsonl(file_path)


class TestDetectFormat:
    """Tests for detect_format function."""

    def test_detect_json(self, tmp_path):
        """Test detection of .json extension."""
        assert detect_format(tmp_path / "file.json") == "json"
        assert detect_format(tmp_path / "file.JSON") == "json"

    def test_detect_jsonl(self, tmp_path):
        """Test detection of .jsonl extension."""
        assert detect_format(tmp_path / "file.jsonl") == "jsonl"
        assert detect_format(tmp_path / "file.JSONL") == "jsonl"

    def test_detect_csv(self, tmp_path):
        """Test detection of .csv extension."""
        assert detect_format(tmp_path / "file.csv") == "csv"
        assert detect_format(tmp_path / "file.CSV") == "csv"

    def test_detect_unsupported(self, tmp_path):
        """Test that ValueError is raised for unsupported formats."""
        with pytest.raises(ValueError, match="Unsupported file format"):
            detect_format(tmp_path / "file.txt")


class TestLoadDataFile:
    """Tests for load_data_file function."""

    def test_load_json_file(self, tmp_path):
        """Test auto-detection and loading of JSON files."""
        data = {"test": "data"}
        file_path = tmp_path / "test.json"
        file_path.write_text(json.dumps(data))

        result = load_data_file(file_path)
        assert result == data

    def test_load_jsonl_file(self, tmp_path):
        """Test auto-detection and loading of JSONL files."""
        file_path = tmp_path / "test.jsonl"
        file_path.write_text('{"id": 1}\n{"id": 2}\n')

        result = load_data_file(file_path)
        assert result == [{"id": 1}, {"id": 2}]

    def test_load_csv_file(self, tmp_path):
        """Test auto-detection and loading of CSV files."""
        file_path = tmp_path / "test.csv"
        with file_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "name"])
            writer.writeheader()
            writer.writerow({"id": "1", "name": "Alice"})
            writer.writerow({"id": "2", "name": "Bob"})

        result = load_data_file(file_path)
        assert len(result) == 2
        assert result[0] == {"id": "1", "name": "Alice"}

    def test_load_file_not_found(self, tmp_path):
        """Test that FileNotFoundError is raised for missing files."""
        with pytest.raises(FileNotFoundError):
            load_data_file(tmp_path / "nonexistent.json")


class TestDatasetLoader:
    """Tests for DatasetLoader class."""

    def test_load_json(self, tmp_path):
        """Test loading JSON with DatasetLoader."""
        data = [{"id": 1}, {"id": 2}]
        file_path = tmp_path / "data.json"
        file_path.write_text(json.dumps(data))

        loader = DatasetLoader(file_path)
        result = loader.load()
        assert result == data

    def test_load_caches_data(self, tmp_path):
        """Test that load() caches the data."""
        file_path = tmp_path / "data.json"
        file_path.write_text('{"test": true}')

        loader = DatasetLoader(file_path)
        result1 = loader.load()
        result2 = loader.load()
        assert result1 is result2

    def test_detect_format(self, tmp_path):
        """Test format detection."""
        file_path = tmp_path / "data.jsonl"
        file_path.write_text('{"id": 1}\n')

        loader = DatasetLoader(file_path)
        assert loader.detect_format() == "jsonl"

    def test_validate_schema_list_success(self, tmp_path):
        """Test schema validation with valid data."""
        data = [
            {"id": 1, "name": "Alice"},
            {"id": 2, "name": "Bob"},
        ]
        file_path = tmp_path / "data.json"
        file_path.write_text(json.dumps(data))

        loader = DatasetLoader(file_path)
        assert loader.validate_schema(["id", "name"]) is True

    def test_validate_schema_list_failure(self, tmp_path):
        """Test schema validation with missing fields."""
        data = [
            {"id": 1, "name": "Alice"},
            {"id": 2},  # Missing "name"
        ]
        file_path = tmp_path / "data.json"
        file_path.write_text(json.dumps(data))

        loader = DatasetLoader(file_path)
        assert loader.validate_schema(["id", "name"]) is False

    def test_validate_schema_dict(self, tmp_path):
        """Test schema validation with dict data."""
        data = {"questions": [], "corpus": []}
        file_path = tmp_path / "data.json"
        file_path.write_text(json.dumps(data))

        loader = DatasetLoader(file_path)
        assert loader.validate_schema(["questions", "corpus"]) is True
        assert loader.validate_schema(["questions", "missing"]) is False

    def test_validate_schema_empty_list(self, tmp_path):
        """Test schema validation with empty list."""
        file_path = tmp_path / "data.json"
        file_path.write_text("[]")

        loader = DatasetLoader(file_path)
        assert loader.validate_schema(["id"]) is True  # Empty list passes


class TestNormalizeText:
    """Tests for normalize_text function."""

    def test_basic_normalization(self):
        """Test basic text normalization."""
        assert normalize_text("  Hello  World  ") == "hello world"

    def test_empty_string(self):
        """Test normalization of empty string."""
        assert normalize_text("") == ""

    def test_unicode_normalization(self):
        """Test Unicode normalization (NFC)."""
        # é as e + combining acute vs precomposed é
        result = normalize_text("café")
        assert result == "café"

    def test_multiple_whitespace(self):
        """Test normalization of multiple whitespace characters."""
        assert normalize_text("hello\t\n  world") == "hello world"

    def test_already_normalized(self):
        """Test that already normalized text remains unchanged."""
        assert normalize_text("hello world") == "hello world"


class TestExtractField:
    """Tests for extract_field function."""

    def test_simple_field(self):
        """Test extracting a simple top-level field."""
        item = {"name": "Alice", "age": 30}
        assert extract_field(item, "name") == "Alice"
        assert extract_field(item, "age") == 30

    def test_nested_field(self):
        """Test extracting a nested field."""
        item = {"user": {"name": "Alice", "email": "alice@example.com"}}
        assert extract_field(item, "user.name") == "Alice"

    def test_deeply_nested_field(self):
        """Test extracting a deeply nested field."""
        item = {"a": {"b": {"c": {"d": "value"}}}}
        assert extract_field(item, "a.b.c.d") == "value"

    def test_list_index_access(self):
        """Test extracting from a list with numeric index."""
        item = {"items": [{"id": 1}, {"id": 2}, {"id": 3}]}
        assert extract_field(item, "items.0.id") == 1
        assert extract_field(item, "items.2.id") == 3

    def test_missing_field(self):
        """Test that missing fields return None."""
        item = {"name": "Alice"}
        assert extract_field(item, "missing") is None
        assert extract_field(item, "name.nested") is None

    def test_missing_nested_field(self):
        """Test that missing nested fields return None."""
        item = {"user": {"name": "Alice"}}
        assert extract_field(item, "user.email") is None
        assert extract_field(item, "nonexistent.field") is None

    def test_none_item(self):
        """Test that None item returns None."""
        assert extract_field(None, "field") is None

    def test_list_index_out_of_range(self):
        """Test that out-of-range list index returns None."""
        item = {"items": [{"id": 1}]}
        assert extract_field(item, "items.5.id") is None

    def test_invalid_list_index(self):
        """Test that non-numeric index on list returns None."""
        item = {"items": [{"id": 1}]}
        assert extract_field(item, "items.name") is None

    def test_object_attribute_access(self):
        """Test extracting from an object with attributes."""

        class Obj:
            name = "test"
            value = 42

        assert extract_field(Obj(), "name") == "test"
        assert extract_field(Obj(), "value") == 42
