"""Example: Chatbot without persistent memory."""
from langgraph.graph import MessagesState
from langgraph.store.base import BaseStore
from langchain_core.runnables import RunnableConfig

from examples.utils import get_llm
from core.graph_factory import create_graph
from memory_strategies import chatbot, NoMemoryStrategy

# Initialize LLM
llm = get_llm()

# Create no-memory strategy
strategy = NoMemoryStrategy()
user_id = "user_123"

def node(state: MessagesState, config: RunnableConfig, *, store: BaseStore):
    """Chatbot node using NoMemoryStrategy."""
    return chatbot(
        state,
        config,
        store=store,
        strategy=strategy,
        llm=llm,
        user_id=user_id
    )

# Create graph without checkpointer or store (stateless)
graph = create_graph(node)
