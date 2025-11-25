# langgraph-kusto

This repository provides Kusto-backed storage and checkpointing for LangGraph, including optional semantic search using Kusto's `ai_embeddings` plugin.

## Kusto store search example

The `KustoStore` supports two search modes:

- **Vector search** when an embedding function is configured
- **Text search fallback** when no embedding function is provided

Below is an example of configuring a `KustoStore` with the `KustoOpenAIEmbeddingFn` and running a search.

```python
from langgraph_kusto.common.kusto_client import KustoClient
from langgraph_kusto.store.store import KustoStore, KustoStoreConfig
from langgraph_kusto.store.embeddings import KustoOpenAIEmbeddingFn

# Create Kusto client from environment
client = KustoClient.from_env()

# Configure embedding function backed by Kusto's ai_embeddings plugin
embedding_fn = KustoOpenAIEmbeddingFn(
    client=client,
    model_uri="https://myaccount.openai.azure.com/openai/deployments/text-embedding-3-small/embeddings?api-version=2024-06-01;impersonate",  # or your configured embedding model
)

# Configure the store
store_config = KustoStoreConfig(
    client=client,
    embedding_function=embedding_fn,
)

store = KustoStore(config=store_config)

# Run a semantic search in a given namespace
results = store.search(namespace="my-namespace", query="find relevant items", k=5)

for item in results:
    print(item["key"], item["score"], item["chunk_string"])
```

If you omit the `embedding_function` when creating `KustoStoreConfig`, `store.search` will automatically fall back to a simple text search over the stored values for the requested `namespace`.
