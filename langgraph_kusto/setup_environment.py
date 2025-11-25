from __future__ import annotations

import os

from .common.kusto_client import KustoClient


def initialize_kusto(
    *,
    client: KustoClient | None = None,
    store_table: str | None = None,
    checkpoints_table: str | None = None,
    embeddings_table: str | None = None,
) -> None:
    if client is None:
        client = KustoClient.from_env()

    # Only read from env if not provided (for backward compatibility)
    if store_table is None:
        store_table = os.getenv("KUSTO_STORE_TABLE", "LangGraphStore")
    if checkpoints_table is None:
        checkpoints_table = os.getenv("KUSTO_CHECKPOINTS_TABLE", "LangGraphCheckpoints")
    if embeddings_table is None:
        embeddings_table = os.getenv("KUSTO_STORE_EMBEDDINGS_TABLE", f"{store_table}Embeddings")

    store_raw = f"{store_table}Raw"
    embeddings_raw = f"{embeddings_table}Raw"
    checkpoints_raw = f"{checkpoints_table}Raw"
    checkpoint_writes_raw = f"{checkpoints_table}WritesRaw"

    store_command = (
        f".create table {store_raw} "
        "(Namespace: string, Key: string, Value: dynamic, CreatedAt: datetime, UpdatedAt: datetime, Tags: dynamic, Deleted: bool)"
    )
    embeddings_command = (
        f".create table {embeddings_raw} "
        "(Namespace: string, ParentKey: string, ChunkOrdinal: long, ChunkString: string, Embedding: dynamic, EmbeddingUri: string, CreatedAt: datetime, Deleted: bool)"
    )
    checkpoints_command = (
        f".create table {checkpoints_raw} "
        "(ThreadId: string, CheckpointNamespace: string, CheckpointId: string, ParentCheckpointId: string, Snapshot: dynamic, CreatedAt: datetime, Deleted: bool)"
    )
    checkpoint_writes_command = (
        f".create table {checkpoint_writes_raw} "
        "(ThreadId: string, CheckpointNamespace: string, CheckpointId: string, TaskId: string, Writes: dynamic, CreatedAt: datetime)"
    )

    store_view = (
        f".create-or-alter function with (folder='langgraph') {store_table}() "
        f"{{ {store_raw} | summarize arg_max(UpdatedAt, *) by Namespace, Key | where Deleted == false }}"
    )
    embeddings_view = (
        f".create-or-alter function with (folder='langgraph') {embeddings_table}() "
        f"{{ {embeddings_raw} | summarize arg_max(CreatedAt, *) by Namespace, ParentKey, ChunkOrdinal | where Deleted == false }}"
    )
    checkpoints_view = (
        f".create-or-alter function with (folder='langgraph') {checkpoints_table}() "
        f"{{ {checkpoints_raw} | summarize arg_max(CreatedAt, *) by ThreadId, CheckpointNamespace, CheckpointId | where Deleted == false }}"
    )
    checkpoint_writes_view = (
        f".create-or-alter function with (folder='langgraph') {checkpoints_table}Writes() "
        f"{{ {checkpoint_writes_raw} }}"
    )

    for cmd_name, cmd in [
        ("store table", store_command),
        ("embeddings table", embeddings_command),
        ("checkpoints table", checkpoints_command),
        ("checkpoint writes table", checkpoint_writes_command),
        ("store view", store_view),
        ("embeddings view", embeddings_view),
        ("checkpoints view", checkpoints_view),
        ("checkpoint writes view", checkpoint_writes_view),
    ]:
        try:
            client.execute_command(cmd)
            print(f"Created Kusto {cmd_name}.")
        except Exception as e:
            print(f"Kusto {cmd_name} creation skipped or failed: {e}")


if __name__ == "__main__":
    initialize_kusto()
