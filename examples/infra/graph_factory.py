"""Graph factory for creating LangGraph workflows."""

from typing import Any, Callable, Optional

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.store.base import BaseStore

# Type alias for a chatbot node function (store is keyword-only)
NodeFunction = Callable[..., dict[str, Any]]


def create_graph(node_fn: NodeFunction, *, checkpoint=None, store: Optional[BaseStore] = None):
    """
    Build and compile a graph with a single chatbot node.

    Args:
        node_fn: The node function with signature (state: MessagesState, config: RunnableConfig, *, store: BaseStore)
        checkpoint: Optional checkpointer for state persistence
        store: Optional store for memory persistence

    Returns:
        Compiled graph ready for invocation
    """
    workflow = StateGraph(MessagesState)
    workflow.add_node("chatbot", node_fn)
    workflow.add_edge(START, "chatbot")
    workflow.add_edge("chatbot", END)
    return workflow.compile(checkpointer=checkpoint, store=store)
