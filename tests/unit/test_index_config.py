from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langgraph.store.base import PutOp

from langgraph_kusto.store.memory_ops import MemoryPut
from langgraph_kusto.store.translator import LanggraphOpToKustoOpTranslator


class TestIndexConfiguration:
    """Test that index configuration is properly passed through the system."""

    def test_translate_put_with_index_none(self):
        """Test that index=None is properly translated."""
        translator = LanggraphOpToKustoOpTranslator()
        op = PutOp(
            namespace=("test",),
            key="k1",
            value={"data": "value"},
            index=None,
        )

        result = translator.translate_op(
            op,
            table_name="TestTable",
            embeddings_table_name="TestEmbeddings",
        )

        assert isinstance(result, MemoryPut)
        assert result.index is None

    def test_translate_put_with_index_false(self):
        """Test that index=False is properly translated."""
        translator = LanggraphOpToKustoOpTranslator()
        op = PutOp(
            namespace=("test",),
            key="k1",
            value={"data": "value"},
            index=False,
        )

        result = translator.translate_op(
            op,
            table_name="TestTable",
            embeddings_table_name="TestEmbeddings",
        )

        assert isinstance(result, MemoryPut)
        assert result.index is False

    def test_translate_put_with_index_paths(self):
        """Test that index with paths is properly translated."""
        translator = LanggraphOpToKustoOpTranslator()
        op = PutOp(
            namespace=("test",),
            key="k1",
            value={"metadata": {"title": "Test"}, "content": "text"},
            index=["metadata.title", "content"],
        )

        result = translator.translate_op(
            op,
            table_name="TestTable",
            embeddings_table_name="TestEmbeddings",
        )

        assert isinstance(result, MemoryPut)
        assert result.index == ["metadata.title", "content"]

    def test_enrichment_with_index_false(self):
        """Test that index=False prevents embedding generation."""
        from langgraph_kusto.store.memory_layer import KustoMemoryLayer

        # Create a mock embedding function
        mock_embedding_fn = MagicMock(return_value=([1.0, 2.0, 3.0], "model-uri"))

        layer = KustoMemoryLayer(embedding_fn=mock_embedding_fn)

        cmd = MemoryPut(
            namespace="test",
            key="k1",
            value={"data": "value"},
            tags=None,
            table_name="TestTable",
            embeddings_table_name="TestEmbeddings",
            namespace_match_type="prefix",
            index=False,
        )

        layer._enrich_command_with_embeddings(cmd)

        # Embedding function should NOT be called
        mock_embedding_fn.assert_not_called()
        assert cmd.embedding_chunks is None

    def test_enrichment_with_index_none(self):
        """Test that index=None embeds the whole value."""
        from langgraph_kusto.store.memory_layer import KustoMemoryLayer

        # Create a mock embedding function
        mock_embedding_fn = MagicMock(return_value=([1.0, 2.0, 3.0], "model-uri"))

        layer = KustoMemoryLayer(embedding_fn=mock_embedding_fn)

        cmd = MemoryPut(
            namespace="test",
            key="k1",
            value={"data": "value"},
            tags=None,
            table_name="TestTable",
            embeddings_table_name="TestEmbeddings",
            namespace_match_type="prefix",
            index=None,
        )

        layer._enrich_command_with_embeddings(cmd)

        # Embedding function should be called once with serialized value
        assert mock_embedding_fn.call_count == 1
        assert cmd.embedding_chunks is not None
        assert len(cmd.embedding_chunks) == 1
        assert cmd.embedding_chunks[0][0] == 0  # ordinal
        assert '"data"' in cmd.embedding_chunks[0][1]  # chunk contains serialized data

    def test_enrichment_with_index_paths(self):
        """Test that index with paths extracts and embeds specific fields."""
        from langgraph_kusto.store.memory_layer import KustoMemoryLayer

        # Create a mock embedding function
        mock_embedding_fn = MagicMock(return_value=([1.0, 2.0, 3.0], "model-uri"))

        layer = KustoMemoryLayer(embedding_fn=mock_embedding_fn)

        cmd = MemoryPut(
            namespace="test",
            key="k1",
            value={
                "metadata": {"title": "Test Title"},
                "content": "Test Content",
            },
            tags=None,
            table_name="TestTable",
            embeddings_table_name="TestEmbeddings",
            namespace_match_type="prefix",
            index=["metadata.title", "content"],
        )

        layer._enrich_command_with_embeddings(cmd)

        # Embedding function should be called twice (once per field)
        assert mock_embedding_fn.call_count == 2
        assert cmd.embedding_chunks is not None
        assert len(cmd.embedding_chunks) == 2

        # Check that the correct values were embedded
        embedded_values = [chunk[1] for chunk in cmd.embedding_chunks]
        assert "Test Title" in embedded_values
        assert "Test Content" in embedded_values

    def test_enrichment_with_wildcard_paths(self):
        """Test that wildcard paths generate multiple embeddings."""
        from langgraph_kusto.store.memory_layer import KustoMemoryLayer

        # Create a mock embedding function
        mock_embedding_fn = MagicMock(return_value=([1.0, 2.0, 3.0], "model-uri"))

        layer = KustoMemoryLayer(embedding_fn=mock_embedding_fn)

        cmd = MemoryPut(
            namespace="test",
            key="k1",
            value={
                "items": [
                    {"text": "Item 1"},
                    {"text": "Item 2"},
                    {"text": "Item 3"},
                ]
            },
            tags=None,
            table_name="TestTable",
            embeddings_table_name="TestEmbeddings",
            namespace_match_type="prefix",
            index=["items[*].text"],
        )

        layer._enrich_command_with_embeddings(cmd)

        # Embedding function should be called 3 times (once per item)
        assert mock_embedding_fn.call_count == 3
        assert cmd.embedding_chunks is not None
        assert len(cmd.embedding_chunks) == 3

        # Check that the correct values were embedded
        embedded_values = [chunk[1] for chunk in cmd.embedding_chunks]
        assert "Item 1" in embedded_values
        assert "Item 2" in embedded_values
        assert "Item 3" in embedded_values
