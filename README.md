# langgraph-kusto

This repository provides Kusto-backed storage and checkpointing for LangGraph, including optional semantic search using Kusto's `ai_embeddings` plugin.

## Kusto store examples

### Configuring the store

The `KustoStore` supports two search modes:

- **Vector search** when an embedding function is configured
- **Text search fallback** when no embedding function is provided

Below is an example of configuring a `KustoStore` with the `KustoOpenAIEmbeddingFn`:

```python
from langgraph_kusto.common.kusto_client import KustoClient
from langgraph_kusto.store.store import KustoStore
from langgraph_kusto.store.config import KustoStoreConfig
from langgraph_kusto.store.embeddings import KustoOpenAIEmbeddingFn

# Create Kusto client from environment
client = KustoClient.from_env()

# Configure embedding function backed by Kusto's ai_embeddings plugin
embedding_fn = KustoOpenAIEmbeddingFn(
    client=client,
    model_uri="https://myaccount.openai.azure.com/openai/deployments/text-embedding-3-small/embeddings?api-version=2024-06-01;impersonate",
)

# Alternative: Use a custom embedding function with Kusto's embedding plugin directly
# This gives you full control over the embedding process
# def custom_embedding_fn(text: str) -> tuple[list[float], str]:
#     """Custom embedding function using Kusto's ai_embeddings plugin."""
#     model_uri = "https://your-endpoint.openai.azure.com/openai/deployments/your-model/embeddings?api-version=2024-06-01"
#     escaped_text = text.replace("'", "''")
#     kql = f"evaluate ai_embeddings('{model_uri}', '{escaped_text}')"
#     result = client.execute_query(kql)
#     embedding = result.primary_results[0][0]["embedding"]
#     return (embedding, model_uri)

# Configure the store
store_config = KustoStoreConfig(
    client=client,
    embedding_function=embedding_fn,
    # embedding_function=custom_embedding_fn,  # Or use your custom function
)

store = KustoStore(config=store_config)
```

If you omit the `embedding_function` when creating `KustoStoreConfig`, `store.search` will automatically fall back to a simple text search over the stored values for the requested `namespace`.

### Inserting and indexing items

You can control which properties of your items are embedded for semantic search using the `index` parameter:

```python
# Insert an item (default: embeds the entire value)
store.put(
    namespace=("documents",),
    key="doc1",
    value={"title": "General Info", "content": "This whole object is embedded."}
)

# Insert an item and embed ONLY specific properties
# Supports dot notation for nested fields (e.g., "metadata.summary")
store.put(
    namespace=("documents",),
    key="doc2",
    value={
        "title": "Specific Embedding",
        "content": "Only this text will be searchable via vector search.",
        "metadata": {"tags": ["hidden"]}
    },
    index=["content"]  # Only embed the "content" field
)

# Disable embedding entirely for an item
store.put(
    namespace=("documents",),
    key="doc3",
    value={"title": "No Embedding", "content": "Not embedded."},
    index=False  # No embeddings created
)
```

### Searching items

Once items are inserted, you can search them using semantic or text-based search:

```python
# Run a semantic search in a given namespace
results = store.search(namespace=("documents",), query="find relevant items", limit=5)

for item in results:
    print(item.key, item.score, item.value)
```
