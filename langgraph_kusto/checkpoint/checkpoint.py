from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import ormsgpack
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
)
from langgraph.checkpoint.serde.jsonplus import _msgpack_ext_hook_to_json

from ..common.kusto_client import KustoClient
from ..store.kql_builder import serialize_value

@dataclass(slots=True)
class KustoCheckpointConfig:
    client: KustoClient
    table_name: str = "LangGraphCheckpoints"

class KustoCheckpointSaver(BaseCheckpointSaver[str]):
    def __init__(self, *, config: KustoCheckpointConfig) -> None:
        super().__init__()
        self._client = config.client
        self._table_name = config.table_name
        self._raw_table_name = f"{config.table_name}Raw"
        self._writes_table_name = f"{config.table_name}Writes"
        self._writes_raw_table_name = f"{config.table_name}WritesRaw"

    def _insert_checkpoint_row(
        self,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
        parent_checkpoint_id: str,
        snapshot: dict,
        deleted: bool = False,
    ) -> None:
        """Insert a row into the raw checkpoint table using set-or-append."""
        created_at = datetime.now(timezone.utc).isoformat()

        command = f""".set-or-append {self._raw_table_name} <|
print 
    ThreadId={serialize_value(thread_id)}, 
    CheckpointNamespace={serialize_value(checkpoint_ns)}, 
    CheckpointId={serialize_value(checkpoint_id)}, 
    ParentCheckpointId={serialize_value(parent_checkpoint_id)}, 
    Snapshot={serialize_value(snapshot)}, 
    CreatedAt=todatetime({serialize_value(created_at)}), 
    Deleted={serialize_value(deleted)}"""

        self._client.execute_command(command)

    def _insert_checkpoint_writes_row(
        self,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
        task_id: str,
        writes: Sequence,
    ) -> None:
        """Insert a row into the raw checkpoint writes table."""
        created_at = datetime.now(timezone.utc).isoformat()

        command = f""".set-or-append {self._writes_raw_table_name} <|
print 
    ThreadId={serialize_value(thread_id)}, 
    CheckpointNamespace={serialize_value(checkpoint_ns)}, 
    CheckpointId={serialize_value(checkpoint_id)}, 
    TaskId={serialize_value(task_id)}, 
    Writes={serialize_value(writes)}, 
    CreatedAt=todatetime({serialize_value(created_at)})"""

        self._client.execute_command(command)

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        thread_id = config["configurable"].get("thread_id")
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = get_checkpoint_id(config)

        if not thread_id:
            return None

        # Build query to get the checkpoint
        if checkpoint_id:
            # Get specific checkpoint
            query = f"""
            {self._table_name}()
            | where ThreadId == '{thread_id}'
                and CheckpointNamespace == '{checkpoint_ns}'
                and CheckpointId == '{checkpoint_id}'
            | take 1
            """
        else:
            # Get latest checkpoint for the thread
            query = f"""
            {self._table_name}()
            | where ThreadId == '{thread_id}'
                and CheckpointNamespace == '{checkpoint_ns}'
            | top 1 by CreatedAt desc
            """

        result = self._client.execute_query(query)
        if not result or not result.primary_results or len(result.primary_results[0]) == 0:
            return None

        row = result.primary_results[0][0]

        # Deserialize the checkpoint using serde
        snapshot_data = row["Snapshot"]
        if isinstance(snapshot_data, str):
            # If it's a string, encode and use serde
            checkpoint = self.serde.loads_typed(("json", snapshot_data.encode("utf-8")))
        elif isinstance(snapshot_data, dict):
            # If it's a dict (from Kusto dynamic), convert to msgpack bytes then use serde
            msgpack_bytes = ormsgpack.packb(snapshot_data)
            checkpoint = self.serde.loads_typed(("msgpack", msgpack_bytes))
        else:
            raise TypeError(f"Unexpected snapshot data type: {type(snapshot_data)}")

        # Query pending writes for this checkpoint
        writes_query = f"""
        {self._writes_table_name}
        | where ThreadId == '{thread_id}'
            and CheckpointNamespace == '{checkpoint_ns}'
            and CheckpointId == '{row["CheckpointId"]}'
        """
        writes_result = self._client.execute_query(writes_query)

        pending_writes = None
        if writes_result and writes_result.primary_results and len(writes_result.primary_results[0]) > 0:
            # Collect all writes from all tasks
            all_writes = []
            for writes_row in writes_result.primary_results[0]:
                writes_data = writes_row["Writes"]
                if writes_data:
                    # Deserialize writes using serde
                    if isinstance(writes_data, str):
                        writes_list = self.serde.loads_typed(("json", writes_data.encode("utf-8")))
                    elif isinstance(writes_data, dict | list):
                        msgpack_bytes = ormsgpack.packb(writes_data)
                        writes_list = self.serde.loads_typed(("msgpack", msgpack_bytes))
                    else:
                        writes_list = writes_data
                    
                    # Convert list of lists back to list of tuples for LangGraph
                    if isinstance(writes_list, list):
                        writes_list = [tuple(item) if isinstance(item, list) else item for item in writes_list]
                    all_writes.extend(writes_list)
            if all_writes:
                pending_writes = all_writes

        # Build metadata
        metadata: CheckpointMetadata = {
            "source": "loop",
            "step": -1,
            "parents": {},
        }
        if row["ParentCheckpointId"]:
            metadata["parents"] = {checkpoint_ns: row["ParentCheckpointId"]}

        # Build configs
        checkpoint_config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": row["CheckpointId"],
            }
        }

        parent_config = None
        if row["ParentCheckpointId"]:
            parent_config = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": row["ParentCheckpointId"],
                }
            }

        return CheckpointTuple(
            config=checkpoint_config,
            checkpoint=checkpoint,
            metadata=metadata,
            parent_config=parent_config,
            pending_writes=pending_writes,
        )

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        thread_id = config["configurable"].get("thread_id") if config else None
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "") if config else ""

        # Build query
        query_parts = [f"{self._table_name}()"]

        # Add filters
        where_clauses = []
        if thread_id:
            where_clauses.append(f"ThreadId == '{thread_id}'")
        if checkpoint_ns:
            where_clauses.append(f"CheckpointNamespace == '{checkpoint_ns}'")

        if where_clauses:
            query_parts.append("| where " + " and ".join(where_clauses))

        # Add before filter
        if before:
            before_checkpoint_id = get_checkpoint_id(before)
            if before_checkpoint_id:
                # Get the timestamp of the "before" checkpoint
                before_query = f"""
                {self._table_name}()
                | where CheckpointId == '{before_checkpoint_id}'
                | project CreatedAt
                | take 1
                """
                before_result = self._client.execute_query(before_query)
                if before_result and before_result.primary_results and len(before_result.primary_results[0]) > 0:
                    before_ts = before_result.primary_results[0][0]["CreatedAt"]
                    query_parts.append(f"| where CreatedAt < datetime({before_ts.isoformat()})")

        # Sort by most recent first
        query_parts.append("| order by CreatedAt desc")

        # Add limit
        if limit:
            query_parts.append(f"| take {limit}")

        query = "\n".join(query_parts)
        result = self._client.execute_query(query)

        if not result or not result.primary_results:
            return

        for row in result.primary_results[0]:
            # Deserialize the checkpoint using serde
            snapshot_data = row["Snapshot"]
            if isinstance(snapshot_data, str):
                checkpoint = self.serde.loads_typed(("json", snapshot_data.encode("utf-8")))
            elif isinstance(snapshot_data, dict):
                msgpack_bytes = ormsgpack.packb(snapshot_data)
                checkpoint = self.serde.loads_typed(("msgpack", msgpack_bytes))
            else:
                checkpoint = snapshot_data

            # Deserialize writes if present using serde
            writes_data = row.get("Writes")
            pending_writes = None
            if writes_data:
                if isinstance(writes_data, str):
                    pending_writes = self.serde.loads_typed(("json", writes_data.encode("utf-8")))
                elif isinstance(writes_data, dict | list):
                    msgpack_bytes = ormsgpack.packb(writes_data)
                    pending_writes = self.serde.loads_typed(("msgpack", msgpack_bytes))
                else:
                    pending_writes = writes_data
                
                # Convert list of lists back to list of tuples for LangGraph
                if isinstance(pending_writes, list):
                    pending_writes = [tuple(item) if isinstance(item, list) else item for item in pending_writes]

            # Build metadata
            metadata: CheckpointMetadata = {
                "source": "loop",
                "step": -1,
                "parents": {},
            }
            if row["ParentCheckpointId"]:
                metadata["parents"] = {row["CheckpointNamespace"]: row["ParentCheckpointId"]}

            # Build configs
            checkpoint_config = {
                "configurable": {
                    "thread_id": row["ThreadId"],
                    "checkpoint_ns": row["CheckpointNamespace"],
                    "checkpoint_id": row["CheckpointId"],
                }
            }

            parent_config = None
            if row["ParentCheckpointId"]:
                parent_config = {
                    "configurable": {
                        "thread_id": row["ThreadId"],
                        "checkpoint_ns": row["CheckpointNamespace"],
                        "checkpoint_id": row["ParentCheckpointId"],
                    }
                }

            yield CheckpointTuple(
                config=checkpoint_config,
                checkpoint=checkpoint,
                metadata=metadata,
                parent_config=parent_config,
                pending_writes=pending_writes,
            )

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = checkpoint["id"]
        parent_checkpoint_id = config["configurable"].get("checkpoint_id", "")

        # Serialize the checkpoint using serde, then convert to dict for Kusto ingestion
        type_, serialized_bytes = self.serde.dumps_typed(checkpoint)
        if type_ == "msgpack":
            # Convert msgpack bytes to JSON-compatible dict via ormsgpack
            # Use the jsonplus ext_hook to convert extensions to plain JSON
            snapshot = ormsgpack.unpackb(
                serialized_bytes, 
                ext_hook=_msgpack_ext_hook_to_json,
                option=ormsgpack.OPT_NON_STR_KEYS
            )
        elif type_ == "json":
            snapshot = json.loads(serialized_bytes.decode("utf-8"))
        elif type_ == "null":
            snapshot = {}
        else:
            # Fallback: try to decode as JSON
            snapshot = json.loads(serialized_bytes.decode("utf-8"))

        self._insert_checkpoint_row(
            thread_id=thread_id,
            checkpoint_ns=checkpoint_ns,
            checkpoint_id=checkpoint_id,
            parent_checkpoint_id=parent_checkpoint_id,
            snapshot=snapshot,
        )

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"].get("checkpoint_id")

        if not checkpoint_id:
            return

        # Serialize the writes using serde, then convert to dict/list for Kusto ingestion
        type_, serialized_bytes = self.serde.dumps_typed(list(writes))
        if type_ == "msgpack":
            # Convert msgpack bytes to JSON-compatible structure via ormsgpack
            serialized_writes = ormsgpack.unpackb(
                serialized_bytes,
                ext_hook=_msgpack_ext_hook_to_json,
                option=ormsgpack.OPT_NON_STR_KEYS
            )
        elif type_ == "json":
            serialized_writes = json.loads(serialized_bytes.decode("utf-8"))
        elif type_ == "null":
            serialized_writes = []
        else:
            # Fallback: try to decode as JSON
            serialized_writes = json.loads(serialized_bytes.decode("utf-8"))

        self._insert_checkpoint_writes_row(
            thread_id=thread_id,
            checkpoint_ns=checkpoint_ns,
            checkpoint_id=checkpoint_id,
            task_id=task_id,
            writes=serialized_writes,
        )

    def delete_thread(self, thread_id: str) -> None:
        # Soft delete: insert a deletion marker
        self._insert_checkpoint_row(
            thread_id=thread_id,
            checkpoint_ns="",
            checkpoint_id="",
            parent_checkpoint_id="",
            snapshot={},
            deleted=True,
        )
