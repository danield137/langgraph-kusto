from __future__ import annotations

import datetime
import json
from typing import Any, Literal

SPECIAL_CHARS = {"\\"}


def _kusto_literal(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        # In KQL, multi-line strings must use triple backticks
        # Also, when potentially troublesome characters are present, just escape with a triple backtick
        if "\n" in value or "\r" in value or any(c in value for c in SPECIAL_CHARS):
            # Triple backtick strings don't need escaping
            return f"```{value}```"
        # Single-line strings use single quotes with doubled quote escaping
        escaped = value.replace("'", "''")
        return f"'{escaped}'"
    if isinstance(value, dict):
        inner = ", ".join(f"{_kusto_literal(k)}: {_kusto_literal(v)}" for k, v in value.items())
        return "{" + inner + "}"
    if isinstance(value, (list, tuple)):
        open_, close_ = ("[", "]") if isinstance(value, list) else ("(", ")")
        body = ", ".join(_kusto_literal(v) for v in value)
        if isinstance(value, tuple) and len(value) == 1:
            body += ","
        return f"{open_}{body}{close_}"
    return repr(value)


def serialize_value(value: Any) -> str:
    """Serialize a Python value to Kusto KQL format."""
    serialized = value
    if isinstance(value, str):
        serialized = f'"{escape_kql_string(value)}"'
    elif isinstance(value, bool):
        # Kusto uses lowercase true/false
        serialized = "true" if value else "false"
    elif isinstance(value, (list, dict)):
        # Serialize lists and dicts as JSON strings
        serialized = f"dynamic({_kusto_literal(value)})"
    elif isinstance(value, datetime.datetime):
        # Format datetime in ISO 8601 format
        serialized = f'datetime("{value.isoformat()}")'
    else:
        serialized = _kusto_literal(value)

    return serialized


def escape_kql_string(value: str) -> str:
    """Escape single quotes for KQL string literals."""
    return value.replace("'", "''")


class KqlBuilder:
    """Builds KQL queries using primitive parameters."""

    @staticmethod
    def memory_get_by_key(
        *, table_name: str, namespace: str, namespace_mode: Literal["prefix", "suffix"], key: str
    ) -> str:
        """Build KQL query to get a single item by key."""
        cond = "startswith" if namespace_mode == "prefix" else "endswith"
        return f"""
{table_name}
| where Namespace {cond} '{namespace}' and Key == '{key}'
| project Value
"""

    @staticmethod
    def memory_get_created_at(
        *, table_name: str, namespace: str, namespace_mode: Literal["prefix", "suffix"], key: str
    ) -> str:
        """Build KQL query to get CreatedAt timestamp for existing item."""
        cond = "startswith" if namespace_mode == "prefix" else "endswith"
        return f"""
{table_name}()
| where Namespace {cond} '{namespace}' and Key == '{key}'
| take 1
| project CreatedAt
"""

    @staticmethod
    def memory_embedding_get_created_at(
        *,
        embeddings_table_name: str,
        namespace: str,
        namespace_mode: Literal["prefix", "suffix"],
        parent_key: str,
        ordinal: int,
    ) -> str:
        """Build KQL query to get CreatedAt timestamp for existing embedding chunk."""
        cond = "startswith" if namespace_mode == "prefix" else "endswith"
        return f"""
{embeddings_table_name}()
| where Namespace {cond} '{namespace}' and ParentKey == '{parent_key}' and ChunkOrdinal == {ordinal}
| take 1
| project CreatedAt
"""

    @staticmethod
    def memory_search_by_similarity(
        *,
        table_name: str,
        embeddings_table_name: str,
        namespace: str,
        namespace_mode: Literal["prefix", "suffix"],
        query_vector: list[float],
        limit: int,
    ) -> str:
        """Build KQL query for vector similarity search."""
        query_vector_json = json.dumps(query_vector)
        cond = "startswith" if namespace_mode == "prefix" else "endswith"

        return f"""
let q = dynamic({query_vector_json});
let store =
    {table_name}
    | where Namespace {cond} '{namespace}';
let emb =
    {embeddings_table_name}
    | where Namespace {cond} '{namespace}';
emb
| extend Score = series_cosine_similarity(Embedding, q)
| top {limit} by Score desc
| summarize arg_max(Score, * ) by ParentKey
| join kind=inner (store) on $left.ParentKey == $right.Key
| project Namespace, Key, Value, Tags, ChunkString, ChunkOrdinal, EmbeddingUri, Score
"""

    @staticmethod
    def memory_search_by_content(
        *, table_name: str, namespace: str, namespace_mode: Literal["prefix", "suffix"], query: str, limit: int
    ) -> str:
        """Build KQL query for text-based content search."""
        query_escaped = query.replace("'", "''")
        cond = "startswith" if namespace_mode == "prefix" else "endswith"

        return f"""
{table_name}()
| where Namespace {cond} '{namespace}'
| where tostring(Value) has '{query_escaped}'
| take {limit}
| project Namespace, Key, Value, Tags, CreatedAt, UpdatedAt
"""

    @staticmethod
    def memory_list_namespaces(*, table_name: str) -> str:
        """Build KQL query to list distinct namespaces."""
        return f"""
{table_name}()
| distinct Namespace
"""
