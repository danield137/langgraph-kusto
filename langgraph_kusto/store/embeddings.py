from __future__ import annotations

from ..common.kusto_client import KustoClient


class KustoOpenAIEmbeddingFn:
    def __init__(self, *, client: KustoClient, model_uri: str) -> None:
        self._client = client
        self._model_uri = model_uri

    def __call__(self, text: str) -> tuple[list[float], str]:
        escaped = text.replace("'", "''")

        kql = f"evaluate ai_embeddings('{self._model_uri}', '{escaped}')"

        result = self._client.execute_query(kql)
        primary = result.primary_results[0]
        rows = list(primary)
        if not rows:
            raise RuntimeError("No embedding returned from ai_embeddings.")

        row = rows[0]
        value = row["embedding"]
        if isinstance(value, list):
            return [float(x) for x in value], self._model_uri

        # Handle dynamic / JSON-encoded embedding
        return [float(x) for x in value], self._model_uri
