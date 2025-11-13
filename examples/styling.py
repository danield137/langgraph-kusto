"""Styling utilities for terminal output using termcolor."""
from termcolor import colored

# Section headers
def banner_line(text: str) -> str:
    """Style a banner line."""
    return colored(text, 'cyan', attrs=['bold'])

def header(text: str) -> str:
    """Style a main header."""
    return colored(text, 'cyan', attrs=['bold'])

def subheader(text: str) -> str:
    """Style a subheader."""
    return colored(text, 'white', attrs=['bold'])

# Mode names
def mode_name(mode_key: str, text: str) -> str:
    """Style mode name based on mode type."""
    colors = {
        "without_memory": ('blue', ['bold']),
        "with_memory": ('green', ['bold']),
        "with_kusto": ('magenta', ['bold']),
    }
    color, attrs = colors.get(mode_key, ('white', ['bold']))
    return colored(text, color, attrs=attrs)

# Status indicators
def status_pass(text: str) -> str:
    """Style a PASS status."""
    return colored(text, 'green', attrs=['bold'])

def status_fail(text: str) -> str:
    """Style a FAIL status."""
    return colored(text, 'red', attrs=['bold'])

def status_skip(text: str) -> str:
    """Style a SKIPPED status."""
    return colored(text, 'yellow', attrs=['bold'])

def status_unexpected(text: str) -> str:
    """Style an UNEXPECTED status."""
    return colored(text, 'red', attrs=['bold', 'underline'])

# Transcript elements
def transcript_label(text: str) -> str:
    """Style a transcript label (User/Agent)."""
    return colored(text, 'white', attrs=['dark'])

def transcript_message(text: str) -> str:
    """Style a transcript message."""
    return colored(text, 'white', attrs=['dark'])

def transcript_reset() -> str:
    """Style the RESET marker."""
    return colored("<< RESET >>", 'yellow', attrs=['dark'])

def transcript_separator() -> str:
    """Style conversation separator."""
    return colored("---", 'white', attrs=['dark'])

# Meta information
def truncate_meta(text: str) -> str:
    """Style truncation metadata."""
    return colored(text, 'white', attrs=['dark'])

def info_text(text: str) -> str:
    """Style informational text."""
    return colored(text, 'white', attrs=['dark'])

# Table elements
def table_header(text: str) -> str:
    """Style table header."""
    return colored(text, 'white', attrs=['bold'])

def table_separator(text: str) -> str:
    """Style table separator."""
    return colored(text, 'white', attrs=['dark'])
