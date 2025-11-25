from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, call

import pytest
from langgraph.store.base import GetOp, ListNamespacesOp, PutOp, SearchOp

from langgraph_kusto.store.config import KustoStoreConfig
from langgraph_kusto.store.kql_builder import KqlBuilder
from langgraph_kusto.store.store import KustoStore


class TestKustoStore:
    """Unit tests for KustoStore with mocked KustoClient."""

    @pytest.fixture
    def mock_client(self):
        """Create a mocked KustoClient."""
        client = MagicMock()
        client.database = "TestDB"
        return client

    @pytest.fixture
    def store(self, mock_client):
        """Create a KustoStore with mocked client (not initialized)."""
        config = KustoStoreConfig(
            client=mock_client,
            table_name="TestStore",
            embeddings_table_name="TestStoreEmbeddings",
        )
        return KustoStore(config=config)

    @pytest.fixture
    def initialized_store(self, store):
        """Pre-initialize a KustoStore."""
        store._initialized = True
        return store

    @pytest.fixture
    def store_with_embeddings(self, mock_client):
        """Create a KustoStore with mocked client and embedding function (not initialized)."""

        def mock_embedding_fn(text: str) -> tuple[list[float], str]:
            # Simple mock: return a fixed-size vector based on text length
            return [float(len(text))] * 384, "mock-model-uri"

        config = KustoStoreConfig(
            client=mock_client,
            table_name="TestStore",
            embeddings_table_name="TestStoreEmbeddings",
            embedding_function=mock_embedding_fn,
        )
        return KustoStore(config=config)

    @pytest.fixture
    def initialized_store_with_embeddings(self, store_with_embeddings):
        """Pre-initialize a KustoStore with embeddings."""
        store_with_embeddings._initialized = True
        return store_with_embeddings

    def _mock_query_result(self, rows: list[dict]):
        """Helper to create a mock query result."""
        result = MagicMock()
        result.primary_results = [rows]
        return result

    def test_initialization(self, store, mock_client):
        """Test 0: Store initialization creates tables and views."""
        # Mock the query response
        mock_client.execute_query.return_value = self._mock_query_result([])

        # Trigger initialization by calling batch
        op = GetOp(namespace=("users",), key="u1")
        store.batch([op])

        # Verify table creation commands
        commands = [call[0][0] for call in mock_client.execute_command.call_args_list]

        # Check for raw table creation
        assert any(".create table TestStoreRaw" in cmd for cmd in commands)
        assert any(".create table TestStoreEmbeddingsRaw" in cmd for cmd in commands)

        # Check for view creation
        assert any(".create-or-alter function" in cmd and "TestStore()" in cmd for cmd in commands)
        assert any(".create-or-alter function" in cmd and "TestStoreEmbeddings()" in cmd for cmd in commands)

    def test_basic_put(self, initialized_store, mock_client):
        """Test 1: Basic put operation generates correct KQL."""
        # Mock the query response (checking for existing CreatedAt)
        mock_client.execute_query.return_value = self._mock_query_result([])

        # Execute put operation
        op = PutOp(namespace=("users", "u1"), key="profile", value={"name": "Alice"})
        initialized_store.batch([op])

        # Verify execute_query was called to check for existing record
        assert mock_client.execute_query.call_count == 1
        query_kql = mock_client.execute_query.call_args[0][0]
        assert "Namespace startswith 'users/u1'" in query_kql
        assert "Key == 'profile'" in query_kql

        # Verify execute_command was called with .set-or-append
        assert mock_client.execute_command.call_count == 1
        command_kql = mock_client.execute_command.call_args[0][0]
        assert ".set-or-append TestStoreRaw <|" in command_kql
        assert 'Namespace="users/u1"' in command_kql
        assert 'Key="profile"' in command_kql
        assert "'name': 'Alice'" in command_kql
        assert "Deleted=false" in command_kql

    def test_put_with_embeddings(self, initialized_store_with_embeddings, mock_client):
        """Test 2: Put with embeddings generates KQL for both tables."""
        # Mock responses
        mock_client.execute_query.return_value = self._mock_query_result([])

        # Execute put operation
        op = PutOp(namespace=("users", "u1"), key="bio", value={"name": "Alice"})
        initialized_store_with_embeddings.batch([op])

        # Should call execute_query twice (once for main record, once for embedding)
        assert mock_client.execute_query.call_count == 2

        # Should call execute_command twice (once for main table, once for embeddings)
        assert mock_client.execute_command.call_count == 2

        # First command: main table
        main_command = mock_client.execute_command.call_args_list[0][0][0]
        assert ".set-or-append TestStoreRaw <|" in main_command
        assert 'Namespace="users/u1"' in main_command

        # Second command: embeddings table
        emb_command = mock_client.execute_command.call_args_list[1][0][0]
        assert ".set-or-append TestStoreEmbeddingsRaw <|" in emb_command
        assert 'Namespace="users/u1"' in emb_command
        assert 'ParentKey="bio"' in emb_command
        assert "Embedding=" in emb_command
        assert "ChunkOrdinal=0" in emb_command

    def test_get_item(self, initialized_store, mock_client):
        """Test 3: Get operation generates correct KQL."""
        # Mock the query response
        mock_row = MagicMock()
        mock_row.to_dict.return_value = {
            "Namespace": ["users", "u1"],
            "Key": "bio",
            "Value": '{"name": "Alice"}',
            "CreatedAt": datetime.now(timezone.utc),
            "UpdatedAt": datetime.now(timezone.utc),
        }
        mock_client.execute_query.return_value = self._mock_query_result([mock_row])

        # Execute get operation
        op = GetOp(namespace=("users", "u1"), key="bio")
        results = initialized_store.batch([op])

        # Verify query matches expected KQL from KqlBuilder
        assert mock_client.execute_query.call_count == 1
        expected_kql = KqlBuilder.memory_get_by_key(
            table_name="TestStore",
            namespace="users/u1",
            namespace_mode="prefix",
            key="bio",
        )
        actual_kql = mock_client.execute_query.call_args[0][0]
        assert actual_kql == expected_kql

        # Verify result
        assert results[0] is not None
        assert results[0].key == "bio"
        assert results[0].namespace == ("users", "u1")
        assert results[0].value == {"name": "Alice"}

    def test_vector_search(self, initialized_store_with_embeddings, mock_client):
        """Test 4: Vector search generates KQL with similarity calculation."""
        # Mock search results
        mock_row = MagicMock()
        mock_row.to_dict.return_value = {
            "Namespace": "users",
            "Key": "u1",
            "Value": '{"name": "Alice"}',
            "Tags": "{}",
            "Score": 0.95,
        }
        mock_client.execute_query.return_value = self._mock_query_result([mock_row])

        # Execute search
        op = SearchOp(namespace_prefix=("users",), query="find Alice", limit=10, offset=0)
        results = initialized_store_with_embeddings.batch([op])

        # Verify query was called
        assert mock_client.execute_query.call_count == 1
        query_kql = mock_client.execute_query.call_args[0][0]

        # Verify KQL contains vector search elements
        assert "series_cosine_similarity" in query_kql
        assert "Namespace startswith 'users'" in query_kql
        assert "top 10 by Score desc" in query_kql

        # Verify results
        assert len(results) == 1
        assert results[0][0].key == "u1"
        assert results[0][0].score == 0.95

    def test_soft_delete(self, initialized_store, mock_client):
        """Test 5: Deleting (putting None) sets Deleted=true."""
        # Execute delete operation (put with None value)
        op = PutOp(namespace=("users",), key="u1", value=None)
        initialized_store.batch([op])

        # Verify execute_command was called
        assert mock_client.execute_command.call_count == 1
        command_kql = mock_client.execute_command.call_args[0][0]

        # Verify soft delete markers
        assert ".set-or-append TestStoreRaw <|" in command_kql
        assert 'Namespace="users"' in command_kql
        assert 'Key="u1"' in command_kql
        assert "Deleted=true" in command_kql
        assert "Value=dynamic({})" in command_kql or "Value=null" in command_kql

    def test_text_search_no_embeddings(self, initialized_store, mock_client):
        """Test 6: Text search without embeddings uses 'has' operator."""
        # Mock search results
        mock_row = MagicMock()
        mock_row.to_dict.return_value = {
            "Namespace": "users",
            "Key": "u1",
            "Value": '{"name": "Alice"}',
            "Tags": "{}",
        }
        mock_client.execute_query.return_value = self._mock_query_result([mock_row])

        # Execute search
        op = SearchOp(namespace_prefix=("users",), query="Alice", limit=10, offset=0)
        results = initialized_store.batch([op])

        # Verify query was called
        assert mock_client.execute_query.call_count == 1
        query_kql = mock_client.execute_query.call_args[0][0]

        # Verify KQL uses text search (not vector)
        assert "has 'Alice'" in query_kql or 'has "Alice"' in query_kql
        assert "Namespace startswith 'users'" in query_kql
        assert "series_cosine_similarity" not in query_kql

        # Verify results
        assert len(results) == 1
        assert results[0][0].key == "u1"
        assert results[0][0].score is None  # Text search doesn't have scores

    def test_list_namespaces(self, initialized_store, mock_client):
        """Test 7: List namespaces generates correct KQL."""
        # Mock namespace results
        mock_row1 = MagicMock()
        mock_row1.to_dict.return_value = {"Namespace": "users"}
        mock_row2 = MagicMock()
        mock_row2.to_dict.return_value = {"Namespace": "posts"}
        mock_client.execute_query.return_value = self._mock_query_result([mock_row1, mock_row2])

        # Execute list namespaces
        op = ListNamespacesOp(match_conditions=None, max_depth=None, limit=100, offset=0)
        results = initialized_store.batch([op])

        # Verify query was called
        assert mock_client.execute_query.call_count == 1
        query_kql = mock_client.execute_query.call_args[0][0]

        # Verify KQL gets distinct namespaces
        assert "distinct Namespace" in query_kql

        # Verify results
        assert len(results) == 1
        assert ("users",) in results[0]
        assert ("posts",) in results[0]

    def test_put_with_multi_path_index(self, initialized_store_with_embeddings, mock_client):
        """Test 8: Put with multiple index paths including wildcards extracts and embeds each field."""
        # Mock responses for CreatedAt checks
        mock_client.execute_query.return_value = self._mock_query_result([])

        # Execute put operation with mixed index paths (nested field + wildcard array)
        op = PutOp(
            namespace=("products",),
            key="product123",
            value={
                "metadata": {
                    "title": "Wireless Headphones - Premium Sound",
                    "category": "Electronics",
                    "brand": "AudioTech",
                },
                "description": "High-quality wireless headphones with noise cancellation",
                "tags": ["wireless", "noise-canceling", "bluetooth", "premium"],
                "reviews": [
                    {"rating": 5, "text": "Great sound quality!"},
                    {"rating": 4, "text": "Comfortable and good battery life"},
                ],
            },
            index=["metadata.title", "tags[*]"],
        )
        initialized_store_with_embeddings.batch([op])

        # Should call execute_query 6 times:
        # 1. Check for existing main record CreatedAt
        # 2. Check for embedding chunk 0 (metadata.title)
        # 3-6. Check for embedding chunks 1-4 (4 tags)
        assert mock_client.execute_query.call_count == 6

        # Should call execute_command twice (main table + embeddings)
        assert mock_client.execute_command.call_count == 2

        # First command: main table
        main_command = mock_client.execute_command.call_args_list[0][0][0]
        assert ".set-or-append TestStoreRaw <|" in main_command
        assert 'Namespace="products"' in main_command
        assert 'Key="product123"' in main_command

        # Second command: embeddings table
        emb_command = mock_client.execute_command.call_args_list[1][0][0]
        assert ".set-or-append TestStoreEmbeddingsRaw <|" in emb_command
        assert 'Namespace="products"' in emb_command
        assert 'ParentKey="product123"' in emb_command

        # Should contain 5 chunks total: 1 for title + 4 for tags
        assert "ChunkOrdinal=0" in emb_command
        assert "ChunkOrdinal=1" in emb_command
        assert "ChunkOrdinal=2" in emb_command
        assert "ChunkOrdinal=3" in emb_command
        assert "ChunkOrdinal=4" in emb_command

        # Should contain the title
        assert "Wireless Headphones - Premium Sound" in emb_command

        # Should contain all tags
        assert "wireless" in emb_command
        assert "noise-canceling" in emb_command
        assert "bluetooth" in emb_command
        assert "premium" in emb_command

        # Should NOT contain description or reviews (not indexed)
        # Note: These might appear in main table command but not in individual chunks
        emb_chunks = emb_command.split("ChunkString=")
        # First split is before any chunks
        for chunk in emb_chunks[1:]:
            # Extract chunk content (between quotes or up to comma)
            if "'" in chunk:
                chunk_content = chunk.split("'")[1]
            else:
                chunk_content = chunk.split(",")[0]
            # Verify this chunk doesn't contain unindexed content
            assert "High-quality wireless" not in chunk_content or "Wireless Headphones" in chunk_content
            assert "Great sound quality" not in chunk_content
