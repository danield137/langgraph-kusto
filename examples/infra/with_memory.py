"""Example: Chatbot with InMemoryStore persistent memory."""

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

from examples.infra.config import get_llm
from examples.infra.graph_factory import create_graph
from examples.memory_strategies import KeywordMemoryStrategy, chatbot

# Initialize LLM and store
llm = get_llm()
store = InMemoryStore()

# Create memory strategy
strategy = KeywordMemoryStrategy()
user_id = "user_123"


def node(state: MessagesState, config: RunnableConfig, *, store: BaseStore):
    """Chatbot node using KeywordMemoryStrategy."""
    return chatbot(state, config, store=store, strategy=strategy, llm=llm, user_id=user_id)


# Create graph with checkpointer and store
checkpointer = MemorySaver()
graph = create_graph(node, checkpoint=checkpointer, store=store)
