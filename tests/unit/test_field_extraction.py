from __future__ import annotations

import pytest

from langgraph_kusto.store.memory_layer import KustoMemoryLayer


class TestJSONPathParsing:
    """Test JSON path parsing functionality."""

    def test_parse_simple_field(self):
        path = "field"
        keys = KustoMemoryLayer._parse_json_path(path)
        assert keys == ["field"]

    def test_parse_nested_field(self):
        path = "parent.child.grandchild"
        keys = KustoMemoryLayer._parse_json_path(path)
        assert keys == ["parent", "child", "grandchild"]

    def test_parse_array_index(self):
        path = "array[0]"
        keys = KustoMemoryLayer._parse_json_path(path)
        assert keys == ["array", 0]

    def test_parse_array_wildcard(self):
        path = "array[*]"
        keys = KustoMemoryLayer._parse_json_path(path)
        assert keys == ["array", "*"]

    def test_parse_negative_index(self):
        path = "array[-1]"
        keys = KustoMemoryLayer._parse_json_path(path)
        assert keys == ["array", -1]

    def test_parse_complex_path(self):
        path = "context[*].content"
        keys = KustoMemoryLayer._parse_json_path(path)
        assert keys == ["context", "*", "content"]

    def test_parse_nested_array_path(self):
        path = "sections[*].paragraphs[*].text"
        keys = KustoMemoryLayer._parse_json_path(path)
        assert keys == ["sections", "*", "paragraphs", "*", "text"]


class TestJSONPathTraversal:
    """Test JSON path traversal functionality."""

    def test_traverse_simple_field(self):
        data = {"field": "value"}
        keys = ["field"]
        result = KustoMemoryLayer._traverse_json_path(data, keys)
        assert result == ["value"]

    def test_traverse_nested_field(self):
        data = {"parent": {"child": {"grandchild": "value"}}}
        keys = ["parent", "child", "grandchild"]
        result = KustoMemoryLayer._traverse_json_path(data, keys)
        assert result == ["value"]

    def test_traverse_array_index(self):
        data = {"array": ["a", "b", "c"]}
        keys = ["array", 0]
        result = KustoMemoryLayer._traverse_json_path(data, keys)
        assert result == ["a"]

    def test_traverse_array_negative_index(self):
        data = {"array": ["a", "b", "c"]}
        keys = ["array", -1]
        result = KustoMemoryLayer._traverse_json_path(data, keys)
        assert result == ["c"]

    def test_traverse_array_wildcard(self):
        data = {"array": [{"id": 1}, {"id": 2}, {"id": 3}]}
        keys = ["array", "*", "id"]
        result = KustoMemoryLayer._traverse_json_path(data, keys)
        assert result == [1, 2, 3]

    def test_traverse_nested_wildcards(self):
        data = {
            "sections": [
                {"paragraphs": [{"text": "p1"}, {"text": "p2"}]},
                {"paragraphs": [{"text": "p3"}, {"text": "p4"}]},
            ]
        }
        keys = ["sections", "*", "paragraphs", "*", "text"]
        result = KustoMemoryLayer._traverse_json_path(data, keys)
        assert result == ["p1", "p2", "p3", "p4"]

    def test_traverse_missing_field(self):
        data = {"field": "value"}
        keys = ["missing"]
        result = KustoMemoryLayer._traverse_json_path(data, keys)
        assert result == []

    def test_traverse_out_of_bounds_index(self):
        data = {"array": ["a", "b"]}
        keys = ["array", 5]
        result = KustoMemoryLayer._traverse_json_path(data, keys)
        assert result == []


class TestFieldExtraction:
    """Test field extraction functionality."""

    def test_extract_simple_field(self):
        data = {"title": "My Title"}
        paths = ["title"]
        result = KustoMemoryLayer._extract_fields(data, paths)
        assert result == [("title", "My Title")]

    def test_extract_nested_field(self):
        data = {"metadata": {"title": "My Title"}}
        paths = ["metadata.title"]
        result = KustoMemoryLayer._extract_fields(data, paths)
        assert result == [("metadata.title", "My Title")]

    def test_extract_array_elements(self):
        data = {"context": [{"content": "c1"}, {"content": "c2"}]}
        paths = ["context[*].content"]
        result = KustoMemoryLayer._extract_fields(data, paths)
        assert result == [
            ("context[*].content", "c1"),
            ("context[*].content", "c2"),
        ]

    def test_extract_multiple_paths(self):
        data = {
            "metadata": {"title": "My Title"},
            "authors": [{"name": "Alice"}],
        }
        paths = ["metadata.title", "authors[0].name"]
        result = KustoMemoryLayer._extract_fields(data, paths)
        assert result == [
            ("metadata.title", "My Title"),
            ("authors[0].name", "Alice"),
        ]

    def test_extract_complex_object(self):
        data = {"config": {"settings": {"enabled": True, "count": 42}}}
        paths = ["config.settings"]
        result = KustoMemoryLayer._extract_fields(data, paths)
        assert len(result) == 1
        assert result[0][0] == "config.settings"
        # The value should be JSON-serialized
        import json

        assert json.loads(result[0][1]) == {"enabled": True, "count": 42}
