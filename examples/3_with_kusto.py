from utils import check_terms_in_response
from models import TestCase

def run_example(test_cases: list[TestCase]) -> tuple[bool, list[list[tuple[str, str]]], dict[str, any]]:
    """
    Run conversation with Kusto-backed memory store for persistent memory.
    Memory persists across thread resets using Azure Data Explorer.
    
    Args:
        test_cases: List of test cases to run
    
    Returns:
        Tuple of:
        - passed: Boolean indicating if tests passed (remembered)
        - conversations: List of conversation histories
        - stats: Dictionary with metrics
    """
    raise NotImplementedError("Kusto memory mode is not yet implemented. Coming soon!")
