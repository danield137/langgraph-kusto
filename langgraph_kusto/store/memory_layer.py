from __future__ import annotations

import datetime
import json
import re
from typing import Any, Callable, cast

from ..common import utc_now
from ..common.kusto_client import KustoClient
from .config import EmbeddingFunction
from .kql_builder import KqlBuilder, escape_kql_string, serialize_value
from .memory_ops import MemoryGet, MemoryListNamespaces, MemoryOp, MemoryPut, MemorySearch


class KustoMemoryLayer:
    """Executes memory commands by generating KQL and calling the Kusto client.

    Responsibilities:
    - Execute abstract memory ops against Kusto (Get, Put, Search, List namespaces)
    - Handle embedding generation when embedding_fn is provided
    - Manage dual-table pattern (raw table + embeddings table)
    - Preserve CreatedAt semantics for both raw data and embedding chunks
    """

    def __init__(self, *, embedding_fn: EmbeddingFunction | None = None) -> None:
        """Initialize the Kusto Memory Layer.

        Parameters:
            embedding_fn: Optional function to generate embeddings from content.
                         Takes content and returns (vector, metadata) tuple.
        """
        self._embedding_fn = embedding_fn

    @staticmethod
    def _parse_json_path(path: str) -> list[str | int]:
        """Parse a JSON path string into a list of keys and indices.

        Supports:
        - Simple fields: "field"
        - Nested fields: "parent.child"
        - Array indexing: "array[0]", "array[-1]", "array[*]"

        Examples:
        - "metadata.title" -> ["metadata", "title"]
        - "context[*].content" -> ["context", "*", "content"]
        - "authors[0].name" -> ["authors", 0, "name"]
        """
        keys: list[str | int] = []
        for segment in path.split("."):
            match = re.match(r"(.+)\[(.*)\]$", segment)
            if match:
                name, index = match.groups()
                keys.append(name)
                if index == "*":
                    keys.append("*")
                else:
                    keys.append(int(index))
            else:
                keys.append(segment)
        return keys

    @staticmethod
    def _traverse_json_path(data: Any, keys: list[str | int]) -> list[Any]:
        """Traverse a data structure using parsed JSON path keys.

        Returns a list of values found at the specified path.
        Handles wildcards (*) by expanding to all array elements.
        """
        if not keys:
            return [data]

        key = keys[0]
        remaining = keys[1:]

        if key == "*":
            if not isinstance(data, list):
                return []
            results: list[Any] = []
            for item in data:
                results.extend(KustoMemoryLayer._traverse_json_path(item, remaining))
            return results

        if isinstance(key, int):
            if not isinstance(data, list):
                return []
            try:
                item = data[key]
                return KustoMemoryLayer._traverse_json_path(item, remaining)
            except IndexError:
                return []

        # key is string
        if not isinstance(data, dict):
            return []

        if key not in data:
            return []

        return KustoMemoryLayer._traverse_json_path(data[key], remaining)

    @staticmethod
    def _extract_fields(value: Any, paths: list[str]) -> list[tuple[str, str]]:
        """Extract field values from a data structure using JSON paths.

        Returns a list of (path, serialized_value) tuples.
        Each path may produce multiple values if wildcards are used.
        """
        results: list[tuple[str, str]] = []

        for path in paths:
            keys = KustoMemoryLayer._parse_json_path(path)
            values = KustoMemoryLayer._traverse_json_path(value, keys)

            for extracted_value in values:
                if isinstance(extracted_value, str):
                    serialized = extracted_value
                else:
                    serialized = json.dumps(extracted_value)
                results.append((path, serialized))

        return results

    def _enrich_command_with_embeddings(self, command: MemoryPut | MemorySearch) -> None:
        """Enrich Put and Search commands with embeddings when embedding_fn is available.

        For MemoryPut:
        - Handles indexing configuration (None, False, or list[str])
        - Extracts field values based on index configuration
        - Calls embedding_fn to get vector + metadata for each field
        - Populates embedding_chunks with (ordinal, chunk_string, vector)
        - Sets embedding_model_uri from metadata

        For MemorySearch:
        - Calls embedding_fn on the query string
        - Populates query_vector for similarity search
        """
        if self._embedding_fn is None:
            return

        if isinstance(command, MemoryPut):
            # Handle index configuration
            if command.index is False:
                # Explicitly disabled indexing
                return

            if command.index is None:
                # Default behavior: embed the whole value
                serialized_value = json.dumps(command.value)
                (vector, metadata) = self._embedding_fn(serialized_value)
                command.embedding_chunks = [(0, serialized_value, vector)]
                command.embedding_model_uri = metadata

            elif isinstance(command.index, list):
                # Extract and embed specific fields
                field_values = self._extract_fields(command.value, command.index)

                chunks: list[tuple[int, str, list[float]]] = []
                metadata = None

                for ordinal, (path, serialized_field_value) in enumerate(field_values):
                    (vector, meta) = self._embedding_fn(serialized_field_value)
                    chunks.append((ordinal, serialized_field_value, vector))
                    if metadata is None:
                        metadata = meta

                command.embedding_chunks = chunks
                command.embedding_model_uri = metadata

        elif isinstance(command, MemorySearch) and command.query:
            (vector, _) = self._embedding_fn(command.query)
            command.query_vector = vector

    def _ingest_rows(self, client: KustoClient, table: str, rows: list[dict]) -> None:
        """Ingest rows into Kusto table using .set-or-append command."""
        if not rows:
            return

        # Build the print statements for each row
        print_statements = []
        for row in rows:
            # Build column=value pairs using Kusto serialization
            columns = ", ".join([f"{key}={serialize_value(value)}" for key, value in row.items()])
            print_statements.append(f"print {columns}")

        # Join all print statements with union
        union_query = " | union ".join(print_statements)

        # Execute the .set-or-append command
        command = f".set-or-append {table} <| {union_query}"
        client.execute_command(command)

    async def _ingest_rows_async(self, client: KustoClient, table: str, rows: list[dict]) -> None:
        """Ingest rows into Kusto table using .set-or-append command asynchronously."""
        if not rows:
            return

        # Build the print statements for each row
        print_statements = []
        for row in rows:
            # Build column=value pairs using Kusto serialization
            columns = ", ".join([f"{key}={serialize_value(value)}" for key, value in row.items()])
            print_statements.append(f"print {columns}")

        # Join all print statements with union
        union_query = " | union ".join(print_statements)

        # Execute the .set-or-append command
        command = f".set-or-append {table} <| {union_query}"
        await client.execute_command_async(command)

    def execute(self, command: MemoryOp, client: KustoClient) -> Any:
        """Execute a memory command synchronously."""
        # Enrich with embeddings before execution if needed
        if isinstance(command, (MemoryPut, MemorySearch)):
            self._enrich_command_with_embeddings(command)

        if isinstance(command, MemoryGet):
            return self._execute_get(command, client)
        if isinstance(command, MemoryPut):
            return self._execute_put(command, client)
        if isinstance(command, MemorySearch):
            return self._execute_search(command, client)
        if isinstance(command, MemoryListNamespaces):
            return self._execute_list_namespaces(command, client)
        raise TypeError(f"Unsupported command type: {type(command)}")

    async def aexecute(self, command: MemoryOp, client: KustoClient) -> Any:
        """Execute a memory command asynchronously."""
        # Enrich with embeddings before execution if needed
        if isinstance(command, (MemoryPut, MemorySearch)):
            self._enrich_command_with_embeddings(command)

        if isinstance(command, MemoryGet):
            return await self._aexecute_get(command, client)
        if isinstance(command, MemoryPut):
            return await self._aexecute_put(command, client)
        if isinstance(command, MemorySearch):
            return await self._aexecute_search(command, client)
        if isinstance(command, MemoryListNamespaces):
            return await self._aexecute_list_namespaces(command, client)
        raise TypeError(f"Unsupported command type: {type(command)}")

    # ===================== GET =====================

    def _execute_get(self, cmd: MemoryGet, client: KustoClient) -> Any | None:
        """Execute a get command synchronously."""
        query = KqlBuilder.memory_get_by_key(
            namespace_mode=cmd.namespace_match_type,
            table_name=cmd.table_name,
            namespace=cmd.namespace,
            key=cmd.key,
        )
        result = client.execute_query(query)
        table = result.primary_results[0]
        rows = list(table)
        if not rows:
            return None

        row = rows[0]
        if hasattr(row, "to_dict"):
            data = row.to_dict()
        else:
            data = dict(row)

        return data

    async def _aexecute_get(self, cmd: MemoryGet, client: KustoClient) -> Any | None:
        """Execute a get command asynchronously."""
        query = KqlBuilder.memory_get_by_key(
            namespace_mode=cmd.namespace_match_type,
            table_name=cmd.table_name,
            namespace=cmd.namespace,
            key=cmd.key,
        )
        result = await client.execute_query_async(query)
        table = result.primary_results[0]
        rows = list(table)
        if not rows:
            return None

        row = rows[0]
        if hasattr(row, "to_dict"):
            data = row.to_dict()
        else:
            data = dict(row)

        return data

    # ===================== PUT =====================

    def _execute_put(self, cmd: MemoryPut, client: KustoClient) -> None:
        """Execute a put command synchronously."""
        if cmd.value is None:
            now = utc_now()
            raw_table = f"{cmd.table_name}Raw"
            rows = [
                {
                    "Namespace": cmd.namespace,
                    "Key": cmd.key,
                    "Value": {},
                    "CreatedAt": now,
                    "UpdatedAt": now,
                    "Tags": {},
                    "Deleted": True,
                }
            ]
            self._ingest_rows(client, raw_table, rows)
            return None

        self._put_raw(cmd, client)

    async def _aexecute_put(self, cmd: MemoryPut, client: KustoClient) -> None:
        """Execute a put command asynchronously."""
        if cmd.value is None:
            now = utc_now()
            raw_table = f"{cmd.table_name}Raw"
            rows = [
                {
                    "Namespace": cmd.namespace,
                    "Key": cmd.key,
                    "Value": json.dumps(None),
                    "CreatedAt": now,
                    "UpdatedAt": now,
                    "Tags": json.dumps({}),
                    "Deleted": True,
                }
            ]
            await self._ingest_rows_async(client, raw_table, rows)
            return None

        await self._aput_raw(cmd, client)

    def _put_raw(self, cmd: MemoryPut, client: KustoClient) -> None:
        """Put raw data synchronously with embeddings support."""
        now = utc_now()

        created_at = now
        existing_query = KqlBuilder.memory_get_created_at(
            namespace_mode=cmd.namespace_match_type,
            table_name=cmd.table_name,
            namespace=cmd.namespace,
            key=cmd.key,
        )

        result = client.execute_query(existing_query)
        table = result.primary_results[0]
        rows = list(table)
        if rows:
            row = rows[0]
            if hasattr(row, "to_dict"):
                data = row.to_dict()
            else:
                data = dict(row)
            stored_created_at = data.get("CreatedAt")
            if stored_created_at is not None:
                created_at = stored_created_at

        raw_table = f"{cmd.table_name}Raw"
        rows = [
            {
                "Namespace": cmd.namespace,
                "Key": cmd.key,
                "Value": cmd.value,
                "CreatedAt": created_at,
                "UpdatedAt": now,
                "Tags": cmd.tags or {},
                "Deleted": False,
            }
        ]
        self._ingest_rows(client, raw_table, rows)

        self._put_embeddings(cmd, cmd.value, now, client)

    async def _aput_raw(self, cmd: MemoryPut, client: KustoClient) -> None:
        """Put raw data asynchronously with embeddings support."""
        now = utc_now()
        serialized_value = json.dumps(cmd.value)
        serialized_tags = json.dumps(cmd.tags or {})

        created_at = now
        existing_query = KqlBuilder.memory_get_created_at(
            namespace_mode=cmd.namespace_match_type,
            table_name=cmd.table_name,
            namespace=cmd.namespace,
            key=cmd.key,
        )

        result = await client.execute_query_async(existing_query)
        table = result.primary_results[0]
        rows = list(table)
        if rows:
            row = rows[0]
            if hasattr(row, "to_dict"):
                data = row.to_dict()
            else:
                data = dict(row)
            stored_created_at = data.get("CreatedAt")
            if stored_created_at is not None:
                created_at = stored_created_at

        raw_table = f"{cmd.table_name}Raw"
        rows = [
            {
                "Namespace": cmd.namespace,
                "Key": cmd.key,
                "Value": serialized_value,
                "CreatedAt": created_at,
                "UpdatedAt": now,
                "Tags": serialized_tags,
                "Deleted": False,
            }
        ]
        await self._ingest_rows_async(client, raw_table, rows)

        await self._aput_embeddings(cmd, serialized_value, now, client)

    def _put_embeddings(self, cmd: MemoryPut, serialized_value: str, now: Any, client: KustoClient) -> None:
        """Store embeddings synchronously."""
        chunks = cmd.embedding_chunks or []
        if not chunks:
            return

        emb_raw_table = f"{cmd.embeddings_table_name}Raw"
        embedding_rows: list[dict] = []

        for ordinal, chunk_string, vector in chunks:
            chunk_created_at = now
            existing_emb_query = KqlBuilder.memory_embedding_get_created_at(
                namespace_mode=cmd.namespace_match_type,
                embeddings_table_name=cmd.embeddings_table_name,
                namespace=cmd.namespace,
                parent_key=cmd.key,
                ordinal=ordinal,
            )

            emb_result = client.execute_query(existing_emb_query)
            emb_table = emb_result.primary_results[0]
            emb_rows = list(emb_table)
            if emb_rows:
                emb_row = emb_rows[0]
                if hasattr(emb_row, "to_dict"):
                    emb_data = emb_row.to_dict()
                else:
                    emb_data = dict(emb_row)
                stored_chunk_created_at = emb_data.get("CreatedAt")
                if stored_chunk_created_at is not None:
                    chunk_created_at = stored_chunk_created_at

            embedding_rows.append(
                {
                    "Namespace": cmd.namespace,
                    "ParentKey": cmd.key,
                    "ChunkOrdinal": ordinal,
                    "ChunkString": chunk_string.replace('"', "'"),
                    "Embedding": vector,
                    "EmbeddingUri": cmd.embedding_model_uri or "",
                    "CreatedAt": chunk_created_at,
                    "Deleted": False,
                }
            )

        self._ingest_rows(client, emb_raw_table, embedding_rows)

    async def _aput_embeddings(self, cmd: MemoryPut, serialized_value: str, now: Any, client: KustoClient) -> None:
        """Store embeddings asynchronously."""
        chunks = cmd.embedding_chunks or []
        if not chunks:
            return

        emb_raw_table = f"{cmd.embeddings_table_name}Raw"
        embedding_rows: list[dict] = []

        for ordinal, chunk_string, vector in chunks:
            chunk_created_at = now
            existing_emb_query = KqlBuilder.memory_embedding_get_created_at(
                namespace_mode=cmd.namespace_match_type,
                embeddings_table_name=cmd.embeddings_table_name,
                namespace=cmd.namespace,
                parent_key=cmd.key,
                ordinal=ordinal,
            )

            emb_result = await client.execute_query_async(existing_emb_query)
            emb_table = emb_result.primary_results[0]
            emb_rows = list(emb_table)
            if emb_rows:
                emb_row = emb_rows[0]
                if hasattr(emb_row, "to_dict"):
                    emb_data = emb_row.to_dict()
                else:
                    emb_data = dict(emb_row)
                stored_chunk_created_at = emb_data.get("CreatedAt")
                if stored_chunk_created_at is not None:
                    chunk_created_at = stored_chunk_created_at

            vector = cmd.embedding_chunks
            embedding_rows.append(
                {
                    "Namespace": cmd.namespace,
                    "ParentKey": cmd.key,
                    "ChunkOrdinal": ordinal,
                    "ChunkString": chunk_string,
                    "Embedding": vector,
                    "EmbeddingUri": cmd.embedding_model_uri or "",
                    "CreatedAt": chunk_created_at,
                    "Deleted": False,
                }
            )

        await self._ingest_rows_async(client, emb_raw_table, embedding_rows)

    # ===================== SEARCH =====================

    def _execute_search(self, cmd: MemorySearch, client: KustoClient) -> list[dict]:
        """Execute a search command synchronously."""
        query = cmd.query or ""

        if cmd.query_vector is not None and query:
            return self._search_with_embeddings(cmd, query, client)
        return self._search_with_text(cmd, query, client)

    async def _aexecute_search(self, cmd: MemorySearch, client: KustoClient) -> list[dict]:
        """Execute a search command asynchronously."""
        query = cmd.query or ""

        if cmd.query_vector is not None and query:
            return await self._asearch_with_embeddings(cmd, query, client)
        return await self._asearch_with_text(cmd, query, client)

    def _search_with_embeddings(self, cmd: MemorySearch, query: str, client: KustoClient) -> list[dict]:
        """Search using vector embeddings synchronously."""
        query_vector = cmd.query_vector or []

        kql = KqlBuilder.memory_search_by_similarity(
            namespace_mode=cmd.namespace_match_type,
            table_name=cmd.table_name,
            embeddings_table_name=cmd.embeddings_table_name,
            namespace=cmd.namespace,
            query_vector=query_vector,
            limit=cmd.limit,
        )

        result = client.execute_query(kql)
        table = result.primary_results[0]
        rows = list(table)

        items: list[dict] = []
        for row in rows[cmd.offset : cmd.offset + cmd.limit]:
            if hasattr(row, "to_dict"):
                data = row.to_dict()
            else:
                data = dict(row)

            items.append(data)

        return items

    async def _asearch_with_embeddings(self, cmd: MemorySearch, query: str, client: KustoClient) -> list[dict]:
        """Search using vector embeddings asynchronously."""
        query_vector = cmd.query_vector or []
        kql = KqlBuilder.memory_search_by_similarity(
            namespace_mode=cmd.namespace_match_type,
            table_name=cmd.table_name,
            embeddings_table_name=cmd.embeddings_table_name,
            namespace=cmd.namespace,
            query_vector=query_vector,
            limit=cmd.limit,
        )

        result = await client.execute_query_async(kql)
        table = result.primary_results[0]
        rows = list(table)

        items: list[dict] = []
        for row in rows[cmd.offset : cmd.offset + cmd.limit]:
            if hasattr(row, "to_dict"):
                data = row.to_dict()
            else:
                data = dict(row)

            items.append(data)

        return items

    def _search_with_text(self, cmd: MemorySearch, query: str, client: KustoClient) -> list[dict]:
        """Search using text matching synchronously."""
        query_escaped = escape_kql_string(query)
        kql = KqlBuilder.memory_search_by_content(
            table_name=cmd.table_name,
            namespace=cmd.namespace,
            namespace_mode=cmd.namespace_match_type,
            query=query_escaped,
            limit=cmd.limit,
        )

        result = client.execute_query(kql)
        table = result.primary_results[0]
        rows = list(table)

        items: list[dict] = []
        for row in rows[cmd.offset : cmd.offset + cmd.limit]:
            if hasattr(row, "to_dict"):
                data = row.to_dict()
            else:
                data = dict(row)

            items.append(data)

        return items

    async def _asearch_with_text(self, cmd: MemorySearch, query: str, client: KustoClient) -> list[dict]:
        """Search using text matching asynchronously."""
        query_escaped = escape_kql_string(query)
        kql = KqlBuilder.memory_search_by_content(
            table_name=cmd.table_name,
            namespace=cmd.namespace,
            namespace_mode=cmd.namespace_match_type,
            query=query_escaped,
            limit=cmd.limit,
        )

        result = await client.execute_query_async(kql)
        table = result.primary_results[0]
        rows = list(table)

        items: list[dict] = []
        for row in rows[cmd.offset : cmd.offset + cmd.limit]:
            if hasattr(row, "to_dict"):
                data = row.to_dict()
            else:
                data = dict(row)

            items.append(data)

        return items

    # ===================== LIST NAMESPACES =====================

    def _execute_list_namespaces(self, cmd: MemoryListNamespaces, client: KustoClient) -> list[str]:
        """Execute a list namespaces command synchronously."""
        kql = f"""
{cmd.table_name}()
| distinct Namespace
"""

        result = client.execute_query(kql)
        table = result.primary_results[0]
        rows = list(table)

        namespaces: list[str] = []
        for row in rows:
            if hasattr(row, "to_dict"):
                data = row.to_dict()
            else:
                data = dict(row)
            ns = data.get("Namespace")
            if ns:
                namespaces.append(ns)

        return namespaces[cmd.offset : cmd.offset + cmd.limit]

    async def _aexecute_list_namespaces(self, cmd: MemoryListNamespaces, client: KustoClient) -> list[str]:
        """Execute a list namespaces command asynchronously."""
        kql = f"""
{cmd.table_name}()
| distinct Namespace
"""

        result = await client.execute_query_async(kql)
        table = result.primary_results[0]
        rows = list(table)

        namespaces: list[str] = []
        for row in rows:
            if hasattr(row, "to_dict"):
                data = row.to_dict()
            else:
                data = dict(row)
            ns = data.get("Namespace")
            if ns:
                namespaces.append(ns)

        return namespaces[cmd.offset : cmd.offset + cmd.limit]
