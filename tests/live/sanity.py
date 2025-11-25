from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # noqa: E402
dotenv.load_dotenv()
import requests

# We keep OpenAIEmbeddings imported for other use cases, but for this
# live test we will call LM Studio's embeddings endpoint directly.
from langchain_openai import OpenAIEmbeddings

from langgraph_kusto.checkpoint.checkpoint import KustoCheckpointConfig, KustoCheckpointSaver
from langgraph_kusto.common import KustoConfig
from langgraph_kusto.common.kusto_client import KustoClient
from langgraph_kusto.store.store import KustoStore, KustoStoreConfig

dotenv.load_dotenv("test.env")


def purge_kusto_resources(
    *,
    client: KustoClient | None = None,
    store_table: str | None = None,
    checkpoints_table: str | None = None,
    embeddings_table: str | None = None,
) -> None:
    """Mirror image of initialize_kusto - drops all tables and functions."""
    if client is None:
        client = KustoClient.from_env()

    if store_table is None:
        store_table = os.getenv("KUSTO_STORE_TABLE", "LangGraphStore")
    if checkpoints_table is None:
        checkpoints_table = os.getenv("KUSTO_CHECKPOINTS_TABLE", "LangGraphCheckpoints")
    if embeddings_table is None:
        embeddings_table = os.getenv("KUSTO_STORE_EMBEDDINGS_TABLE", f"{store_table}Embeddings")

    store_raw = f"{store_table}Raw"
    embeddings_raw = f"{embeddings_table}Raw"
    checkpoints_raw = f"{checkpoints_table}Raw"
    checkpoints_writes_raw = f"{checkpoints_table}WritesRaw"

    # Drop functions first (they depend on tables)
    drop_commands = [
        ("store view", f".drop function {store_table} ifexists"),
        ("embeddings view", f".drop function {embeddings_table} ifexists"),
        ("checkpoints view", f".drop function {checkpoints_table} ifexists"),
        ("checkpoint writes view", f".drop function {checkpoints_table}Writes ifexists"),
        ("store table", f".drop table {store_raw} ifexists"),
        ("embeddings table", f".drop table {embeddings_raw} ifexists"),
        ("checkpoints table", f".drop table {checkpoints_raw} ifexists"),
        ("checkpoint writes table", f".drop table {checkpoints_writes_raw} ifexists"),
    ]

    for cmd_name, cmd in drop_commands:
        try:
            client.execute_command(cmd)
            print(f"Dropped Kusto {cmd_name}.")
        except Exception as e:
            print(f"Kusto {cmd_name} drop skipped or failed: {e}")


def lmstudio_embedding_fn(text: Any) -> tuple[list[float], str]:
    """Call LM Studio's embeddings endpoint for nomic-embed-text-v1.5.

    This is a simple, direct client tailored for LM Studio's OpenAI-compatible
    embeddings API. It returns a list[float] as expected by KustoStore.
    """

    # Normalize to a plain string
    if not isinstance(text, str):
        text = json.dumps(text, ensure_ascii=False)

    base_url = os.getenv("EMBEDDING_BASE_URL", "http://localhost:1234/v1")
    api_key = os.getenv("EMBEDDING_API_KEY", "lm-studio")
    model = os.getenv("EMBEDDING_MODEL", "text-embedding-nomic-embed-text-v1.5")

    url = f"{base_url.rstrip('/')}/embeddings"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "input": text,
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    # Standard OpenAI-style response: { "data": [ { "embedding": [...] } ], ... }
    embedding = data["data"][0]["embedding"]
    return [float(x) for x in embedding], base_url + "/embeddings" + f"?model={model}"


def _wrap_openai_embedding_function(embeddings: OpenAIEmbeddings) -> Callable[[Any], list[float]]:
    """Adapter to ensure OpenAIEmbeddings.embed_query always receives a string.

    The Kusto executor passes JSON-serialized values as chunk strings. This wrapper
    normalizes any input to a plain string before calling `embed_query`, keeping
    OpenAIEmbeddings as the underlying client.
    """

    def _inner(value: Any) -> list[float]:
        if isinstance(value, str):
            text = value
        else:
            import json

            text = json.dumps(value, ensure_ascii=False)

        return embeddings.embed_query(text)

    # Optionally expose model identifier for debugging or downstream usage
    try:
        model = getattr(embeddings, "model", None)
        if model is not None:
            setattr(_inner, "_model_uri", str(model))
    except Exception:
        pass

    return _inner


def _build_store(*, embedding_function: Callable[[Any], tuple[list[float], str]] | None = None) -> KustoStore:
    cluster_uri = os.getenv("KUSTO_CLUSTER_URI")
    database = os.getenv("KUSTO_DATABASE")

    if not cluster_uri or not database:
        raise RuntimeError("KUSTO_CLUSTER_URI and KUSTO_DATABASE must be set in environment")

    kusto_config = KustoConfig(cluster_uri=cluster_uri, database=database)
    client = KustoClient(config=kusto_config)

    store_config = KustoStoreConfig(
        client=client,
        embedding_function=embedding_function,
    )
    return KustoStore(config=store_config)


def run_happy_flow_test() -> None:
    """
    Happy flow test:
    1. Add memories with different namespaces, keys, tags, times
    2. Query by filter (namespace) and verify
    3. Upsert values and verify updates
    4. Soft-delete and verify removal
    5. Search by similarity (with embeddings)
    6. Search by text (without embeddings)
    """
    print("=" * 80)
    print("HAPPY FLOW LIVE TEST")
    print("=" * 80)

    # Setup: Purge resources
    print("\n[SETUP] Purging Kusto resources...")
    cluster_uri = os.getenv("KUSTO_CLUSTER_URI")
    database = os.getenv("KUSTO_DATABASE")

    if not cluster_uri or not database:
        raise RuntimeError("KUSTO_CLUSTER_URI and KUSTO_DATABASE must be set in environment")

    kusto_config = KustoConfig(cluster_uri=cluster_uri, database=database)
    client = KustoClient(config=kusto_config)
    purge_kusto_resources(client=client)

    # Get embedding configuration (for logging only; LM Studio client uses env vars)
    embedding_base_url = os.getenv("EMBEDDING_BASE_URL", "http://localhost:1234/v1")
    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-nomic-embed-text-v1.5")

    print(f"\n[SETUP] Using LM Studio embedding endpoint at {embedding_base_url} with model {embedding_model}")

    # Initialize store with LM Studio embeddings
    print("\n[SETUP] Initializing KustoStore with LM Studio embeddings...")
    store = _build_store(embedding_function=lmstudio_embedding_fn)

    # Step 1: Add memories
    print("\n[STEP 1] Adding memories with different namespaces, keys, tags...")

    memories = [
        {
            "namespace": ("users", "alice"),
            "key": "profile",
            "value": {"name": "Alice", "role": "engineer", "interests": ["python", "ai"]},
            "tags": ["profile", "user"],
        },
        {
            "namespace": ("users", "bob"),
            "key": "profile",
            "value": {"name": "Bob", "role": "designer", "interests": ["ui", "ux"]},
            "tags": ["profile", "user"],
        },
        {
            "namespace": ("projects", "alpha"),
            "key": "summary",
            "value": {"title": "Project Alpha", "status": "active", "description": "AI research and development"},
            "tags": ["project", "active"],
        },
        {
            "namespace": ("projects", "beta"),
            "key": "summary",
            "value": {
                "title": "Project Beta",
                "status": "planning",
                "description": "Building spaceships with recycled materials",
            },
            "tags": ["project", "planning"],
        },
        {
            "namespace": ("notes", "2024"),
            "key": "jan",
            "value": {"month": "January", "content": "Started new AI initiative"},
            "tags": ["note", "2024"],
        },
    ]

    for mem in memories:
        store.put(mem["namespace"], mem["key"], mem["value"])
        print(f"  Added: {mem['namespace']}/{mem['key']}")

    # Step 2: Query by filter (namespace)
    print("\n[STEP 2] Querying by namespace filter...")

    user_results = store.search(("users",))
    print(f"  Found {len(user_results)} items in 'users' namespace:")
    for item in user_results:
        print(f"    - {item.namespace}/{item.key}: {item.value.get('name', 'N/A')}")

    project_results = store.search(("projects",))
    print(f"  Found {len(project_results)} items in 'projects' namespace:")
    for item in project_results:
        print(f"    - {item.namespace}/{item.key}: {item.value.get('title', 'N/A')}")

    # Step 3: Upsert values
    print("\n[STEP 3] Upserting values...")

    # Update Alice's profile
    updated_value = {"name": "Alice", "role": "senior engineer", "interests": ["python", "ai", "mlops"]}
    store.put(("users", "alice"), "profile", updated_value)
    print("  Updated Alice's profile")

    # Verify the update
    alice_item = store.get(("users", "alice"), "profile")
    if alice_item and alice_item.value.get("role") == "senior engineer":
        print(f"  ✓ Verified: Alice's role is now '{alice_item.value.get('role')}'")
    else:
        print("  ✗ Failed to verify update")

    # Step 4: Soft-delete
    print("\n[STEP 4] Soft-deleting item...")

    item_to_delete = ("notes", "2024"), "jan"
    # verify it exists before deletion
    existing_item = store.get(("notes", "2024"), "jan")
    if existing_item is None:
        print(" ✗ Verified: Item doesn't exist before deletion")

    store.delete(("notes", "2024"), "jan")
    print("  Deleted: notes/2024/jan")

    # Verify deletion
    deleted_item = store.get(("notes", "2024"), "jan")
    if deleted_item is None:
        print("  ✓ Verified: Item no longer returned by get()")
    else:
        print("  ✗ Failed: Item still returned by get()")

    # Step 5: Search by similarity
    print("\n[STEP 5] Searching by similarity (with embeddings)...")

    similarity_query = "aliens"
    similarity_results = store.search(("projects",), query=similarity_query, limit=5)
    print(f"  Query: '{similarity_query}'")
    print(f"  Found {len(similarity_results)} results:")
    for item in similarity_results:
        print(f"    - {item.namespace}/{item.key}: {item.value.get('name', 'N/A')} (score: {item.score:.4f})")
        print(f"      Description: {item.value.get('description', 'N/A')}")

    # Step 6: Text-based search (without embeddings)
    print("\n[STEP 6] Re-initializing store without embeddings for text search...")

    store_no_embeddings = _build_store(embedding_function=None)

    text_query = "designer"
    text_results = store_no_embeddings.search(("users",), query=text_query, limit=5)
    print(f"  Query: '{text_query}'")
    print(f"  Found {len(text_results)} results:")
    for item in text_results:
        print(f"    - {item.namespace}/{item.key}: {item.value.get('name', 'N/A')}")
        print(f"      Role: {item.value.get('role', 'N/A')}")

    print("\n" + "=" * 80)
    print("HAPPY FLOW TEST COMPLETED")
    print("=" * 80)


def run_checkpoint_test() -> None:
    """
    Checkpoint test covering the writes flow:
    1. Create checkpoint with initial state
    2. Put writes (with tuples that will be converted to lists in Kusto)
    3. Get checkpoint and verify pending_writes are converted back to tuples
    4. List checkpoints and verify tuple conversion
    5. Verify tuple/list round-trip works correctly with LangGraph
    """
    print("=" * 80)
    print("CHECKPOINT WRITES LIVE TEST")
    print("=" * 80)

    # Setup
    print("\n[SETUP] Setting up checkpoint test...")
    cluster_uri = os.getenv("KUSTO_CLUSTER_URI")
    database = os.getenv("KUSTO_DATABASE")

    if not cluster_uri or not database:
        raise RuntimeError("KUSTO_CLUSTER_URI and KUSTO_DATABASE must be set in environment")

    kusto_config = KustoConfig(cluster_uri=cluster_uri, database=database)
    client = KustoClient(config=kusto_config)

    # Initialize checkpoint saver
    checkpoint_config = KustoCheckpointConfig(client=client)
    saver = KustoCheckpointSaver(config=checkpoint_config)

    print("  ✓ Initialized KustoCheckpointSaver")

    # Step 1: Create a checkpoint with initial state
    print("\n[STEP 1] Creating checkpoint with initial state...")

    thread_id = "test-thread-writes"
    checkpoint_ns = ""
    checkpoint_id = "checkpoint-001"

    config = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
        }
    }

    checkpoint = {
        "v": 1,
        "id": checkpoint_id,
        "ts": datetime.now(timezone.utc).isoformat(),
        "channel_values": {
            "messages": ["Hello", "World"],
            "context": {"user": "test"},
        },
        "channel_versions": {},
        "versions_seen": {},
        "pending_sends": [],
    }

    metadata = {
        "source": "loop",
        "step": 0,
        "parents": {},
    }

    # Put the checkpoint
    returned_config = saver.put(config, checkpoint, metadata, {})
    print(f"  ✓ Created checkpoint: {checkpoint_id}")
    print(f"  Returned config: {returned_config}")

    # Step 2: Put writes (with tuples)
    print("\n[STEP 2] Putting writes (tuples will be converted to lists for Kusto)...")

    # These writes contain tuples: (channel_name, value)
    writes = [
        ("messages", "How are you?"),
        ("messages", "I'm doing great!"),
        ("context", {"updated": True, "timestamp": datetime.now(timezone.utc).isoformat()}),
    ]

    write_config = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
            "checkpoint_id": checkpoint_id,
        }
    }

    saver.put_writes(write_config, writes, task_id="task-1")
    print(f"  ✓ Put {len(writes)} writes")
    for i, (channel, value) in enumerate(writes):
        print(f"    {i + 1}. Channel '{channel}': {value}")

    # Step 3: Get checkpoint and verify pending_writes
    print("\n[STEP 3] Getting checkpoint and verifying pending_writes are tuples...")

    retrieved_tuple = saver.get_tuple(write_config)

    if retrieved_tuple is None:
        print("  ✗ Failed: Could not retrieve checkpoint")
        return

    print(f"  ✓ Retrieved checkpoint: {retrieved_tuple.config['configurable']['checkpoint_id']}")

    if retrieved_tuple.pending_writes is None:
        print("  ✗ Failed: No pending_writes found")
        return

    print(f"  ✓ Found {len(retrieved_tuple.pending_writes)} pending writes")

    # Verify each write is a tuple (not a list)
    all_tuples = True
    for i, write in enumerate(retrieved_tuple.pending_writes):
        is_tuple = isinstance(write, tuple)
        symbol = "✓" if is_tuple else "✗"
        print(f"    {symbol} Write {i + 1}: type={type(write).__name__}, value={write}")
        if not is_tuple:
            all_tuples = False

    if all_tuples:
        print("  ✓ All pending_writes are tuples (correctly converted from Kusto lists)")
    else:
        print("  ✗ Some pending_writes are not tuples!")

    # Step 4: Verify tuple structure matches original (with normalization for Kusto round-trip)
    print("\n[STEP 4] Verifying tuple structure matches original writes...")

    def normalize_for_comparison(value):
        """Normalize values for comparison, accounting for Kusto serialization quirks."""
        if isinstance(value, str):
            # Kusto may strip apostrophes from strings
            normalized = value.replace("'", "")
            # Normalize timezone format: +00:00 -> Z
            normalized = normalized.replace("+00:00", "Z")
            # Normalize timestamp precision (Kusto may add trailing zero to microseconds)
            # e.g., "2025-11-25T06:51:06.180462Z" vs "2025-11-25T06:51:06.1804620Z"
            import re

            # Match ISO timestamps and normalize microsecond precision to 6 digits
            timestamp_pattern = r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6})\d*Z"
            normalized = re.sub(timestamp_pattern, r"\1Z", normalized)
            return normalized
        elif isinstance(value, dict):
            # Recursively normalize dict values
            return {k: normalize_for_comparison(v) for k, v in value.items()}
        elif isinstance(value, list):
            # Recursively normalize list items
            return [normalize_for_comparison(item) for item in value]
        elif isinstance(value, tuple):
            # Recursively normalize tuple items
            return tuple(normalize_for_comparison(item) for item in value)
        return value

    matches = True
    for i, (original, retrieved) in enumerate(zip(writes, retrieved_tuple.pending_writes)):
        # Normalize both for comparison
        normalized_original = normalize_for_comparison(original)
        normalized_retrieved = normalize_for_comparison(retrieved)

        if normalized_original == normalized_retrieved:
            print(f"  ✓ Write {i + 1} matches: {retrieved}")
        else:
            print(f"  ✗ Write {i + 1} mismatch!")
            print(f"    Original:  {original}")
            print(f"    Retrieved: {retrieved}")
            print(f"    Normalized Original:  {normalized_original}")
            print(f"    Normalized Retrieved: {normalized_retrieved}")
            matches = False

    if matches:
        print("  ✓ All writes match original structure (after normalization)")
    else:
        print("  ✗ Some writes don't match!")

    # Step 5: List checkpoints and verify tuple conversion
    print("\n[STEP 5] Listing checkpoints and verifying tuple conversion...")

    list_config = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
        }
    }

    checkpoints = list(saver.list(list_config, limit=10))
    print(f"  ✓ Found {len(checkpoints)} checkpoint(s)")

    for i, cp_tuple in enumerate(checkpoints):
        if cp_tuple.pending_writes:
            print(f"  Checkpoint {i + 1}:")
            for j, write in enumerate(cp_tuple.pending_writes):
                is_tuple = isinstance(write, tuple)
                symbol = "✓" if is_tuple else "✗"
                print(f"    {symbol} Write {j + 1}: type={type(write).__name__}")

    # Step 6: Clean up - delete thread
    print("\n[STEP 6] Cleaning up...")
    saver.delete_thread(thread_id)
    print(f"  ✓ Deleted thread: {thread_id}")

    print("\n" + "=" * 80)
    print("CHECKPOINT WRITES TEST COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    run_happy_flow_test()
    print("\n")
    run_checkpoint_test()
