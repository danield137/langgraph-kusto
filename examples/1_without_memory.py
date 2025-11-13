from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import HumanMessage

from utils import get_llm, check_terms_in_response
from models import TestCase

def run_example(test_cases: list[TestCase]) -> tuple[bool, list[list[tuple[str, str]]], dict[str, any]]:
    """
    Run conversation without persistent memory.
    No checkpointer means no memory between invocations.
    
    Args:
        test_cases: List of test cases to run
    
    Returns:
        Tuple of:
        - passed: Boolean indicating if tests passed (forgot as expected)
        - conversations: List of conversation histories
        - stats: Dictionary with metrics
    """
    llm = get_llm()
    
    workflow = StateGraph(MessagesState)
    
    def chatbot(state: MessagesState):
        """Simple chatbot node"""
        response = llm.invoke(state["messages"])
        return {"messages": [response]}
    
    workflow.add_node("chatbot", chatbot)
    workflow.add_edge(START, "chatbot")
    workflow.add_edge("chatbot", END)
    
    graph = workflow.compile()
    
    conversations = []
    matches = []
    total_forgot = 0
    
    for test_case in test_cases:
        conversation_history = []
        
        # Each invocation is completely independent
        for user_msg in test_case.messages:
            response = graph.invoke(
                {"messages": [HumanMessage(content=user_msg)]}
            )
            agent_reply = response["messages"][-1].content
            conversation_history.append(("user", user_msg))
            conversation_history.append(("agent", agent_reply))
        
        # RESET marker for clarity
        conversation_history.append(("RESET", ""))
        
        # Ask recall question - will have no context
        response = graph.invoke(
            {"messages": [HumanMessage(content=test_case.question)]}
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
        
        if not found:
            total_forgot += 1
    
    # For "without memory", success means forgetting
    passed = total_forgot == len(test_cases)
    accuracy = total_forgot / len(test_cases) if test_cases else 0.0
    
    stats = {
        "accuracy": accuracy,
        "total_tests": len(test_cases),
        "forgot_count": total_forgot,
        "remembered_count": len(test_cases) - total_forgot,
        "matches": matches
    }
    
    return passed, conversations, stats
