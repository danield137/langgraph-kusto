# Field Extraction and Indexing Configuration Implementation

## Overview

This document describes the implementation of field extraction and indexing configuration support for the LangGraph Kusto store, enabling selective field embedding as specified in the LangGraph store documentation.

## Problem Statement

The previous implementation always embedded the entire item value as a single vector. According to the LangGraph store specification, the `index` parameter in `PutOp` should control how items are indexed:

- `None` (default): Use store's default indexing (whole item)
- `False`: Disable indexing for this item
- `list[str]`: Specify which JSON path fields to index for search

## Changes Made

### 1. Updated `MemoryPut` Dataclass (`langgraph_kusto/store/memory_ops.py`)

Added `index` field to support indexing configuration:

```python
@dataclass(slots=True)
class MemoryPut(MemoryNamespacedOp):
    # ... existing fields ...
    index: list[str] | None | Literal[False] = None
    embedding_chunks: list[tuple[int, str, list[float]]] | None = None
    embedding_model_uri: str | None = None
```

### 2. Updated Translator (`langgraph_kusto/store/translator.py`)

Modified `translate_op` to pass the `index` field from `PutOp` to `MemoryPut`:

```python
if isinstance(op, PutOp):
    return MemoryPut(
        # ... existing parameters ...
        index=op.index,
    )
```

### 3. Implemented Field Extraction Logic (`langgraph_kusto/store/memory_layer.py`)

Added three new static methods to `KustoMemoryLayer`:

#### `_parse_json_path(path: str) -> list[str | int]`

Parses JSON path strings into a list of keys and indices. Supports:
- Simple fields: `"field"`
- Nested fields: `"parent.child.grandchild"`
- Array indexing: `"array[0]"` (specific index), `"array[-1]"` (last element), `"array[*]"` (all elements)

#### `_traverse_json_path(data: Any, keys: list[str | int]) -> list[Any]`

