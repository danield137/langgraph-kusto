from __future__ import annotations

from typing import Iterable

from langgraph.store.base import BaseStore, Op, Result

from langgraph_kusto.store.config import KustoStoreConfig
from langgraph_kusto.store.memory_layer import KustoMemoryLayer
from langgraph_kusto.store.translator import LanggraphOpToKustoOpTranslator

from ..setup_environment import initialize_kusto


class KustoStore(BaseStore):
    def __init__(self, *, config: KustoStoreConfig) -> None:
        self._client = config.client
        self._table_name = config.table_name
        self._embeddings_table_name = config.embeddings_table_name
        self._initialized = False

        self._translator = LanggraphOpToKustoOpTranslator()
        self._memory = KustoMemoryLayer(embedding_fn=config.embedding_function)

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return

        initialize_kusto(
            client=self._client,
            store_table=self._table_name,
            embeddings_table=self._embeddings_table_name,
        )

        self._initialized = True

    async def _a_ensure_initialized(self) -> None:
        if self._initialized:
            return

        initialize_kusto(
            client=self._client,
            store_table=self._table_name,
            embeddings_table=self._embeddings_table_name,
        )

        self._initialized = True

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        self._ensure_initialized()
        results: list[Result] = []

        for op in ops:
            command = self._translator.translate_op(
                op,
                table_name=self._table_name,
                embeddings_table_name=self._embeddings_table_name,
            )

            raw_result = self._memory.execute(command, self._client)
            result = self._translator.translate_result(raw_result, op)
            results.append(result)

        return results

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        await self._a_ensure_initialized()
        results: list[Result] = []

        for op in ops:
            command = self._translator.translate_op(
                op,
                table_name=self._table_name,
                embeddings_table_name=self._embeddings_table_name,
            )

            raw_result = await self._memory.aexecute(command, self._client)
            result = self._translator.translate_result(raw_result, op)
            results.append(result)

        return results
