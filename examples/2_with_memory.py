from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langchain_core.messages import HumanMessage, SystemMessage
import uuid

from utils import get_llm, check_terms_in_response
from models import TestCase
from memory_strategies import KeywordMemoryStrategy

def run_example(test_cases: list[TestCase]) -> tuple[bool, list[list[tuple[str, str]]], dict[str, any]]:
    """
    Run conversation with InMemoryStore for persistent memory.
    Memory persists across thread resets.
    
    Args:
        test_cases: List of test cases to run
    
    Returns:
        Tuple of:
        - passed: Boolean indicating if tests passed (remembered)
        - conversations: List of conversation histories
        - stats: Dictionary with metrics
    """
    llm = get_llm()
    store = InMemoryStore()
    user_id = "user_123"
    strategy = KeywordMemoryStrategy()
    
    workflow = StateGraph(MessagesState)
    
    def chatbot_with_memory(state: MessagesState):
        """Chatbot that uses KeywordMemoryStrategy for memory"""
        messages = state["messages"]
        
        # Recall memories using strategy
        memory_items = strategy.recall(store=store, user_id=user_id, messages=messages)
        
        # Build context from memories
        invoke_messages = messages
        if memory_items:
            lines = [f"- {m.get('content', '')}" for m in memory_items]
            memory_context = (
                "You are a helpful assistant.\n\n"
                "IMPORTANT - Remember these facts about the user:\n" +
                "\n".join(lines) +
                "\n\nUse this information when answering questions about their preferences."
            )
            invoke_messages = [SystemMessage(content=memory_context)] + messages
        
        response = llm.invoke(invoke_messages)
        
        # Store important information in memory
        last_user_msg = messages[-1] if messages else None
        strategy.remember(
            store=store,
            user_id=user_id,
            last_user_msg=last_user_msg,
            messages=messages
        )
        
        return {"messages": [response]}
    
    workflow.add_node("chatbot", chatbot_with_memory)
    workflow.add_edge(START, "chatbot")
    workflow.add_edge("chatbot", END)
    
    checkpointer = MemorySaver()
    graph = workflow.compile(checkpointer=checkpointer)
    
    conversations = []
    matches = []
    total_remembered = 0
    
    for test_case in test_cases:
        # First conversation
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        
        conversation_history = []
        for user_msg in test_case.messages:
            response = graph.invoke(
                {"messages": [HumanMessage(content=user_msg)]},
                config=config
            )
            agent_reply = response["messages"][-1].content
            conversation_history.append(("user", user_msg))
            conversation_history.append(("agent", agent_reply))
        
        # RESET - new thread BUT memory persists
        conversation_history.append(("RESET", ""))
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        
        # Ask recall question
        response = graph.invoke(
            {"messages": [HumanMessage(content=test_case.question)]},
            config=config
        )
        final_reply = response["messages"][-1].content
        conversation_history.append(("user", test_case.question))
        conversation_history.append(("agent", final_reply))
        
        conversations.append(conversation_history)
        
        # Check if terms were found
        found, matched_terms = check_terms_in_response(final_reply, test_case.expected_terms)
        matches.append({
            "test": test_case.question,
            "found": found,
            "matched": matched_terms,
            "reply": final_reply
        })
        
        if found:
            total_remembered += 1
    
    # For "with memory", success means remembering
    passed = total_remembered == len(test_cases)
    accuracy = total_remembered / len(test_cases) if test_cases else 0.0
    
    stats = {
        "accuracy": accuracy,
        "total_tests": len(test_cases),
        "remembered_count": total_remembered,
        "forgot_count": len(test_cases) - total_remembered,
        "matches": matches,
        "store": store
    }
    
    return passed, conversations, stats
