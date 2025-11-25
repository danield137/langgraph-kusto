from dataclasses import dataclass
from typing import Any, Callable

from langgraph_kusto.common.kusto_client import KustoClient

# returns a tuple of (embedding vector, metadata)
EmbeddingFunction = Callable[[Any], tuple[list[float], str]]


@dataclass(slots=True)
class KustoStoreConfig:
    client: KustoClient
    table_name: str = "LangGraphStore"
    embeddings_table_name: str = "LangGraphStoreEmbeddings"
    embedding_function: EmbeddingFunction | None = None
