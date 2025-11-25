from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Literal, cast

from langgraph.store.base import GetOp, Item, ListNamespacesOp, Op, PutOp, Result, SearchItem, SearchOp

from .memory_ops import MemoryGet, MemoryListNamespaces, MemoryOp, MemoryPut, MemorySearch


class LanggraphOpToKustoOpTranslator:
    """Translates LangGraph operations to Kusto commands and results back to LangGraph items."""

    @staticmethod
    def _namespace_to_str(namespace: tuple[str, ...]) -> str:
        """Convert namespace tuple to string representation."""
        return "/".join(namespace)

    @staticmethod
    def _str_to_namespace(namespace: str | None) -> tuple[str, ...]:
        """Convert string namespace to tuple representation."""
        if not namespace:
            return tuple()
        return tuple(str(namespace).split("/"))

    def translate_op(
        self,
        op: Op,
        *,
        table_name: str,
        embeddings_table_name: str,
        namespace_mode: Literal["prefix", "suffix"] = "prefix",
    ) -> MemoryOp:
        """Translate a LangGraph Op to a memory command."""
        if isinstance(op, GetOp):
            return MemoryGet(
                namespace_match_type=namespace_mode,
                namespace=self._namespace_to_str(op.namespace),
                key=op.key,
                table_name=table_name,
            )
        if isinstance(op, PutOp):
            return MemoryPut(
                namespace=self._namespace_to_str(op.namespace),
                key=op.key,
                value=op.value,
                tags=None,
                table_name=table_name,
                embeddings_table_name=embeddings_table_name,
                namespace_match_type=namespace_mode,
                index=op.index,
            )
        if isinstance(op, SearchOp):
            return MemorySearch(
                namespace=self._namespace_to_str(op.namespace_prefix),
                namespace_match_type=namespace_mode,
                query=op.query,
                limit=op.limit,
                offset=op.offset,
                table_name=table_name,
                embeddings_table_name=embeddings_table_name,
            )
        if isinstance(op, ListNamespacesOp):
            return MemoryListNamespaces(
                match_conditions=op.match_conditions,
                max_depth=op.max_depth,
                limit=op.limit,
                offset=op.offset,
                table_name=table_name,
            )
        raise TypeError(f"Unsupported op type: {type(op)}")

    def translate_get_result(self, raw: Any, op: GetOp) -> Item | None:
        """Translate a raw memory get result to a LangGraph Item."""
        if raw is None:
            return None

        # ``raw`` is expected to be a Kusto SDK result; tests treat it as
        # ``result.primary_results[0][0].to_dict()``.
        if isinstance(raw, list) and raw:
            # Assume first row is the payload
            row = raw[0]
            if hasattr(row, "to_dict"):
                row = row.to_dict()
        else:
            row = raw

        if not isinstance(row, dict):
            return None

        value_raw = row.get("Value")
        if isinstance(value_raw, str):
            try:
                value = json.loads(value_raw)
            except Exception:  # pragma: no cover - defensive
                value = {"value": value_raw}
        elif isinstance(value_raw, dict):
            value = cast(dict[str, Any], value_raw)
        else:
            value = {"value": value_raw}

        created_at = row.get("CreatedAt") or datetime.now(timezone.utc)
        updated_at = row.get("UpdatedAt") or created_at

        return Item(
            value=value,
            key=op.key,
            namespace=op.namespace,
            created_at=created_at,
            updated_at=updated_at,
        )

    def translate_put_result(self, raw: Any, op: PutOp) -> None:
        """Translate a raw memory put result (always returns None)."""
        return None

    def translate_search_result(self, raw: list[dict], op: SearchOp) -> list[SearchItem]:
        """Translate raw memory search results to LangGraph SearchItems."""
        items: list[SearchItem] = []
        # ``raw`` is expected to be a list of dicts from ``row.to_dict()``.
        for row in raw:
            ns_tuple = self._str_to_namespace(cast(str | None, row.get("Namespace")))
            value_raw = row.get("Value")
            score = cast(float | None, row.get("Score"))

            if isinstance(value_raw, str):
                try:
                    value = json.loads(value_raw)
                except Exception:  # pragma: no cover - defensive
                    value = {"value": value_raw}
            elif isinstance(value_raw, dict):
                value = cast(dict[str, Any], value_raw)
            else:
                value = {"value": value_raw}

            created_at = row.get("CreatedAt") or datetime.now(timezone.utc)
            updated_at = row.get("UpdatedAt") or created_at

            items.append(
                SearchItem(
                    namespace=ns_tuple,
                    key=cast(str, row.get("Key")),
                    value=value,
                    created_at=created_at,
                    updated_at=updated_at,
                    score=score,
                )
            )

        return items

    def translate_list_namespaces_result(self, raw: list[str], op: ListNamespacesOp) -> list[tuple[str, ...]]:
        """Translate raw memory namespace list to LangGraph namespace tuples."""
        namespaces = [self._str_to_namespace(ns) for ns in raw]

        def matches(ns: tuple[str, ...]) -> bool:
            if op.match_conditions is None:
                return True
            for cond in op.match_conditions:
                path = cond.path
                if cond.match_type == "prefix":
                    if tuple(ns[: len(path)]) != path:
                        return False
                else:
                    if tuple(ns[-len(path) :]) != path:
                        return False
            return True

        filtered: list[tuple[str, ...]] = []
        for ns in namespaces:
            if not matches(ns):
                continue
            if op.max_depth is not None and len(ns) > op.max_depth:
                ns = ns[: op.max_depth]
            filtered.append(ns)

        return filtered

    def translate_result(self, raw: Any, op: Op) -> Result:
        """Translate a raw memory result to the appropriate LangGraph result type."""
        if isinstance(op, GetOp):
            return self.translate_get_result(raw, op)
        if isinstance(op, PutOp):
            return self.translate_put_result(raw, op)
        if isinstance(op, SearchOp):
            return self.translate_search_result(raw, op)
        if isinstance(op, ListNamespacesOp):
            return self.translate_list_namespaces_result(raw, op)
        raise TypeError(f"Unsupported op type: {type(op)}")
