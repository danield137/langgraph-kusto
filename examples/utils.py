from langchain_openai import ChatOpenAI
from examples.styling import (
    transcript_label, transcript_message, transcript_reset, 
    transcript_separator, truncate_meta
)

def get_llm():
    """Create and return LM Studio LLM instance"""
    return ChatOpenAI(
        base_url="http://localhost:1234/v1",
        api_key="lm-studio",
        temperature=0.7,
    )

def format_message(text: str, max_len: int = 50) -> str:
    """Format a message with truncation indicator"""
    if len(text) <= max_len:
        return transcript_message(f'"{text}"')
    truncated = f'"{text[:max_len]}..."'
    meta = truncate_meta(f" (trimmed from {len(text)} chars)")
    return transcript_message(truncated) + meta

def print_conversation(history: list[tuple[str, str]]):
    """Print conversation history in a clean format"""
    print(f"   {transcript_separator()}")
    for role, message in history:
        if role == "RESET":
            print(f"   {transcript_reset()}")
        elif role == "user":
            label = transcript_label("User: ")
            print(f"   {label} {format_message(message)}")
        elif role == "agent":
            label = transcript_label("Agent:")
            print(f"   {label} {format_message(message)}")
    print(f"   {transcript_separator()}")

def check_terms_in_response(response: str, expected_terms: list[str]) -> tuple[bool, list[str]]:
    """
    Check if any of the expected terms appear in the response (case-insensitive).
    
    Returns:
        Tuple of (found, matched_terms)
    """
    response_lower = response.lower() if isinstance(response, str) else ""
    matched_terms = [term for term in expected_terms if term.lower() in response_lower]
    return len(matched_terms) > 0, matched_terms
