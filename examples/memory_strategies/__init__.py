"""Memory strategies for LangGraph chatbots."""

from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import MessagesState
from langgraph.store.base import BaseStore

from .keyword import KeywordMemoryStrategy
from .none import NoMemoryStrategy
from .protocol import MemoryStrategy


def chatbot(
    state: MessagesState, config: RunnableConfig, *, store: BaseStore, strategy: MemoryStrategy, llm, user_id: str
):
    """Generic chatbot node that uses a memory strategy.

    The chatbot follows this flow:
    1. Uses strategy.recall() to retrieve relevant memories
    2. Builds system message with memory context if memories exist
    3. Invokes LLM with context-enhanced messages
    4. Uses strategy.remember() to persist new information

    Args:
        state: Current MessagesState
        config: Runnable configuration
        store: BaseStore for memory persistence
        strategy: Memory strategy implementing recall/remember methods
        llm: Language model to invoke
        user_id: Unique identifier for the user

    Returns:
        Dictionary with updated messages
    """
    messages = state["messages"]

    # Recall memories using strategy
    memory_items = strategy.recall(store=store, user_id=user_id, messages=messages)

    # Build message list with memory context if memories exist
    invoke_messages = messages
    if memory_items:
        lines = [f"- {m.get('content', '')}" for m in memory_items]
        system_message = SystemMessage(
            content=("You are a helpful assistant.\n\n" "Remembered user facts:\n" + "\n".join(lines))
        )
        invoke_messages = [system_message] + messages

    # Invoke LLM
    response = llm.invoke(invoke_messages)

    # Store new information using strategy
    last_user_msg = messages[-1] if messages else None
    strategy.remember(store=store, user_id=user_id, last_user_msg=last_user_msg, messages=messages)

    return {"messages": [response]}


__all__ = ["chatbot", "MemoryStrategy", "KeywordMemoryStrategy", "NoMemoryStrategy"]
