#!/usr/bin/env python3
from __future__ import annotations

import sys
import time
from pathlib import Path

import dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # noqa: E402
dotenv.load_dotenv()

"""
Example demonstrating a chatbot with checkpoint support using Kusto.

This example shows how to use the KustoCheckpointSaver to persist conversation state,
allowing the chatbot to resume conversations across sessions.
"""

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from examples.infra.utils import get_llm
from langgraph_kusto.checkpoint import KustoCheckpointConfig, KustoCheckpointSaver
from langgraph_kusto.common.kusto_client import KustoClient
from langgraph_kusto.setup_environment import initialize_kusto


class State(TypedDict):
    messages: list


def chatbot(state: State) -> dict:
    """Chatbot that uses an LLM to respond to messages."""
    llm = get_llm()
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


def create_graph_with_checkpoint():
    """Create a LangGraph with checkpoint support."""
    # Initialize Kusto client
    client = KustoClient.from_env()
    initialize_kusto(client=client)
    # Create checkpoint config
    checkpoint_config = KustoCheckpointConfig(client=client, table_name="LangGraphCheckpoints")

    # Create checkpoint saver
    checkpointer = KustoCheckpointSaver(config=checkpoint_config)

    # Build the graph
    workflow = StateGraph(State)
    workflow.add_node("chatbot", chatbot)
    workflow.add_edge(START, "chatbot")
    workflow.add_edge("chatbot", END)

    # Compile with checkpointer
    app = workflow.compile(checkpointer=checkpointer)

    return app


def main():
    """Run the chatbot example demonstrating checkpoint persistence."""
    app = create_graph_with_checkpoint()

    # Create a thread ID for the conversation
    thread_id = "thread_" + str(int(time.time()))
    config = {"configurable": {"thread_id": thread_id}}

    print("Chatbot with Checkpoint Support")
    print("=" * 50)
    print(f"Thread ID: {thread_id}")
    print()

    # First message: "1 + 1 = X"
    print("You: 1 + 1 = X")
    result = app.invoke({"messages": [HumanMessage(content="Remember, 1 + 1 = X")]}, config=config)
    bot_message = result["messages"][-1]
    print(f"Bot: {bot_message.content}\n")

    # Reset and reload context from checkpoint
    print("Resetting and reloading context from checkpoint...")
    print()

    # Create a new app instance to simulate reset/reload
    app = create_graph_with_checkpoint()

    # Second message: "X + 2 = ?" (using the same thread_id to reload checkpoint)
    print("You: X + 2 = ?")
    result = app.invoke({"messages": [HumanMessage(content="X + 2 = ?")]}, config=config)
    bot_message = result["messages"][-1]
    print(f"Bot: {bot_message.content}\n")

    print("Conversation saved! The checkpoint was successfully persisted and reloaded.")


if __name__ == "__main__":
    main()
