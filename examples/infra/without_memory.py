"""Example: Chatbot without persistent memory."""

from langchain_core.runnables import RunnableConfig
from langgraph.graph import MessagesState
from langgraph.store.base import BaseStore

from examples.infra.config import get_llm
from examples.infra.graph_factory import create_graph
from examples.memory_strategies import NoMemoryStrategy, chatbot

# Initialize LLM
llm = get_llm()

# Create no-memory strategy
strategy = NoMemoryStrategy()
user_id = "user_123"


def node(state: MessagesState, config: RunnableConfig, *, store: BaseStore):
    """Chatbot node using NoMemoryStrategy."""
    return chatbot(state, config, store=store, strategy=strategy, llm=llm, user_id=user_id)


# Create graph without checkpointer or store (stateless)
graph = create_graph(node)
