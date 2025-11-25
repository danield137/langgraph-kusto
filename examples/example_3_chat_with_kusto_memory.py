#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # noqa: E402
dotenv.load_dotenv()

"""Example: Chatbot with KustoStore persistent memory.

This example demonstrates using KustoStore for persistent memory
while using MemorySaver for in-memory checkpointing.

Configuration via environment variables:
    LLM_PROVIDER: "openai" or "lmstudio" (default: "lmstudio")
    EMBEDDING_PROVIDER: "openai", "lmstudio", or "kusto" (default: "lmstudio")
    
See examples/infra/config.py for full configuration options.
"""

import os
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState
from langgraph.store.base import BaseStore

from examples.infra.config import get_embedding_function, get_llm, print_config
from examples.infra.graph_factory import create_graph
from examples.infra.runner import run_scenarios
from examples.infra.scenario import scenarios
from examples.infra.utils import print_conversation
from examples.memory_strategies import KeywordMemoryStrategy, chatbot
from langgraph_kusto.common import KustoConfig
from langgraph_kusto.common.kusto_client import KustoClient
from langgraph_kusto.store.config import KustoStoreConfig
from langgraph_kusto.store.store import KustoStore


def build_kusto_store() -> KustoStore:
    """Build and configure KustoStore from environment variables."""
    cluster_uri = os.getenv("KUSTO_CLUSTER_URI")
    database = os.getenv("KUSTO_DATABASE")
    if not cluster_uri or not database:
        raise RuntimeError("KUSTO_CLUSTER_URI and KUSTO_DATABASE must be set in environment")

    kusto_config = KustoConfig(cluster_uri=cluster_uri, database=database)
    client = KustoClient(config=kusto_config)

    # Use configurable embedding function
    embedder = get_embedding_function()

    store_config = KustoStoreConfig(
        client=client,
        embedding_function=embedder,
    )
    return KustoStore(config=store_config)


def run_example() -> tuple[bool, list[list[tuple[str, str]]], dict[str, Any]]:
    """
    Run conversation with Kusto-backed memory store for persistent memory.
    Memory persists across thread resets using Azure Data Explorer.

    Returns:
        Tuple of:
        - passed: Boolean indicating if tests passed (remembered)
        - conversations: List of conversation histories
        - stats: Dictionary with metrics
    """
    print("\n📋 Configuration:")
    print_config()
    print()

    # Initialize components
    llm = get_llm()
    store = build_kusto_store()
    checkpointer = MemorySaver()
    strategy = KeywordMemoryStrategy()
    user_id = "user_123"

    def node(state: MessagesState, config: RunnableConfig, *, store: BaseStore):
        """Chatbot node using KeywordMemoryStrategy with KustoStore."""
        return chatbot(state, config, store=store, strategy=strategy, llm=llm, user_id=user_id)

    # Create graph with checkpointer and store
    graph = create_graph(node, checkpoint=checkpointer, store=store)

    # Run scenarios
    passed, conversations, stats = run_scenarios(
        graph,
        scenarios,
        expect_memory=True,
    )

    for i, conversation in enumerate(conversations, 1):
        print_conversation(conversation, scenario_num=i)

    return passed, conversations, stats


if __name__ == "__main__":
    run_example()
