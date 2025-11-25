#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # noqa: E402
dotenv.load_dotenv()

"""Single-mode example: chatbot with persistent memory using InMemoryStore.

This example reuses the shared graph defined in
`examples.infra.with_memory` so that the node signature and
graph construction follow the LangGraph persistence pattern.
"""

from examples.infra import with_memory
from examples.infra.runner import run_scenarios
from examples.infra.scenario import scenarios
from examples.infra.utils import print_conversation


def run_example():
    """Run the "with memory" mode for all scenarios.

    This is a thin wrapper around the shared graph and the
    common test runner used by `run_memory_examples.py`.
    """
    passed, conversations, stats = run_scenarios(
        with_memory.graph,
        scenarios,
        expect_memory=True,
    )

    for i, conversation in enumerate(conversations, 1):
        print_conversation(conversation, scenario_num=i)

    return passed, conversations, stats


if __name__ == "__main__":
    run_example()
