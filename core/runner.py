"""Test runner for LangGraph examples."""
import uuid
from typing import Callable, Optional
from langchain_core.messages import HumanMessage
from examples.utils import check_terms_in_response
from examples.models import TestCase


def run_tests(
    graph,
    test_cases: list[TestCase],
    *,
    expect_memory: bool,
    scorer: Optional[Callable] = None
) -> tuple[bool, list[list[tuple[str, str]]], dict]:
    """
    Run all test_cases through graph. Returns (passed, conversations, stats).
    
    Args:
        graph: Compiled LangGraph to test
        test_cases: List of test cases to run
        expect_memory: Whether to expect the graph to remember (True) or forget (False)
        scorer: Optional custom scoring function (default uses check_terms_in_response)
        
    Returns:
        Tuple of:
        - passed: Boolean indicating if tests passed expectations
        - conversations: List of conversation histories
        - stats: Dictionary with test metrics
    """
    conversations = []
    matches = []
    correct = 0
    
    for test_case in test_cases:
        conversation_history = []
        
        # Create thread config for graphs with checkpointer
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        
        # Run through initial messages
        for user_msg in test_case.messages:
            response = graph.invoke({"messages": [HumanMessage(content=user_msg)]}, config)
            agent_reply = response["messages"][-1].content
            conversation_history.append(("user", user_msg))
            conversation_history.append(("agent", agent_reply))
        
        # RESET marker for readability - new thread simulates conversation reset
        conversation_history.append(("RESET", ""))
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        
        # Ask recall question
        response = graph.invoke({"messages": [HumanMessage(content=test_case.question)]}, config)
        final_reply = response["messages"][-1].content
        conversation_history.append(("user", test_case.question))
        conversation_history.append(("agent", final_reply))
        
        conversations.append(conversation_history)
        
        # Score the response
        if scorer:
            found, matched_terms = scorer(final_reply, test_case.expected_terms)
        else:
            found, matched_terms = check_terms_in_response(final_reply, test_case.expected_terms)
        
        matches.append({
            "test": test_case.question,
            "found": found,
            "matched": matched_terms,
            "reply": final_reply
        })
        
        # Check if result matches expectation
        if (found and expect_memory) or (not found and not expect_memory):
            correct += 1
    
    passed = correct == len(test_cases)
    accuracy = correct / len(test_cases) if test_cases else 0.0
    
    stats = {
        "accuracy": accuracy,
        "total_tests": len(test_cases),
        "correct": correct,
        "forgot_count": len(test_cases) - correct if expect_memory else correct,
        "remembered_count": correct if expect_memory else len(test_cases) - correct,
        "matches": matches
    }
    
    return passed, conversations, stats
