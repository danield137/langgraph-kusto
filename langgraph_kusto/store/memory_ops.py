from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


@dataclass(slots=True)
class MemoryOp: ...


@dataclass(slots=True)
class MemoryNamespacedOp(MemoryOp):
    namespace: str
    namespace_match_type: Literal["prefix", "suffix"]  # 'prefix' or 'suffix'


@dataclass(slots=True)
class MemoryGet(MemoryNamespacedOp):
    """Command to retrieve a single item from memory store."""

    key: str
    table_name: str


@dataclass(slots=True)
class MemoryPut(MemoryNamespacedOp):
    """Command to store or update an item in memory store."""

    key: str
    value: Any
    tags: dict[str, Any] | None
    table_name: str
    embeddings_table_name: str
    index: list[str] | None | Literal[False] = None
    embedding_chunks: list[tuple[int, str, list[float]]] | None = None
    embedding_model_uri: str | None = None


@dataclass(slots=True)
class MemorySearch(MemoryNamespacedOp):
    """Command to search items in memory store."""

    query: str | None
    limit: int
    offset: int
    table_name: str
    embeddings_table_name: str
    query_vector: list[float] | None = None


@dataclass(slots=True)
class MemoryListNamespaces(MemoryOp):
    match_conditions: tuple[Any, ...] | None
    max_depth: int | None
    limit: int
    offset: int
    table_name: str
