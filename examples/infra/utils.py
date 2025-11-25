"""Utility functions for examples."""

from examples.infra.config import get_llm  # Re-export for backwards compatibility
from examples.infra.styling import (
    transcript_label,
    transcript_message,
    transcript_reset,
    transcript_separator,
    truncate_meta,
)

__all__ = ["get_llm", "format_message", "print_conversation", "check_terms_in_response"]


def format_message(text: str, max_len: int = 50) -> str:
    """Format a message with truncation indicator"""
    if len(text) <= max_len:
        return transcript_message(f'"{text}"')
    truncated = f'"{text[:max_len]}..."'
    meta = truncate_meta(f" (trimmed from {len(text)} chars)")
    return transcript_message(truncated) + meta


def print_conversation(history: list[tuple[str, str]], scenario_num: int | None = None):
    """Print conversation history in a clean format"""
    if scenario_num is not None:
        print(f"\n#### START Scenario {scenario_num} ####")
    for role, message in history:
        if role == "RESET":
            print(f"   {transcript_reset()}")
        elif role == "user":
            label = transcript_label("User: ")
            print(f"   {label} {format_message(message)}")
        elif role == "agent":
            label = transcript_label("Agent:")
            print(f"   {label} {format_message(message)}")
    if scenario_num is not None:
        print(f"#### END Scenario {scenario_num} ####")


def check_terms_in_response(response: str, expected_terms: list[str]) -> tuple[bool, list[str]]:
    """
    Check if any of the expected terms appear in the response (case-insensitive).

    Returns:
        Tuple of (found, matched_terms)
    """
    response_lower = response.lower() if isinstance(response, str) else ""
    matched_terms = [term for term in expected_terms if term.lower() in response_lower]
    return len(matched_terms) > 0, matched_terms
