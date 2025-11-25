#!/usr/bin/env python3
"""Run all memory examples comparatively.

This script runs all three memory modes side-by-side to demonstrate
the difference between no memory, in-memory store, and Kusto-backed store.

Usage:
    # With LMStudio (default):
    python run_memory_demo.py

    # With OpenAI:
    LLM_PROVIDER=openai EMBEDDING_PROVIDER=openai python run_memory_demo.py

    # With Kusto embeddings:
    EMBEDDING_PROVIDER=kusto python run_memory_demo.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
dotenv.load_dotenv()

from examples.infra.config import print_config
from examples.infra.models import ModeResult, ScenarioResult
from examples.infra.runner import run_scenarios
from examples.infra.scenario import MODES, scenarios
from examples.infra.styling import (
    banner_line,
    info_text,
    mode_name,
    status_fail,
    status_pass,
    status_skip,
    status_unexpected,
    subheader,
    table_separator,
)
from examples.infra.utils import print_conversation


def load_example_modules():
    """Dynamically load all example_*.py modules from the examples directory."""
    modules = {}
    examples_dir = Path(__file__).parent / "examples"

    for filepath in sorted(examples_dir.glob("example_*.py")):
        # Extract module key from filename: example_1_without_memory.py -> without_memory
        # Or: example_3_with_kusto_memory.py -> with_kusto_memory
        stem = filepath.stem  # e.g., "example_1_without_memory"
        parts = stem.split("_", 2)  # ["example", "1", "without_memory"]

        if len(parts) >= 3:
            module_key = parts[2]  # "without_memory", "with_memory", "with_kusto_memory"
        else:
            module_key = stem

        try:
            spec = importlib.util.spec_from_file_location(stem, filepath)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                modules[module_key] = module
                print(f"  ✓ Loaded {filepath.name} as '{module_key}'")
        except Exception as e:
            print(f"  ✗ Could not load {filepath.name}: {e}")

    return modules


def run_all_examples():
    """Run all memory examples and compare results."""
    print("\n" + banner_line("=" * 70))
    print(banner_line("🤖 LangGraph Memory Demo - Comparative Run"))
    print(banner_line("=" * 70))

    print("\n📋 Configuration:")
    print_config()

    print(f"\n{subheader('📊 Running')} {len(scenarios)} {subheader('test case(s)...')}")

    example_modules = load_example_modules()
    results = []

    for i, scenario in enumerate(scenarios, 1):
        if i > 1:
            print("\n" + table_separator("=" * 70))

        test_header = f"📝 Test #{i}: {' → '.join(scenario.messages)}"
        print(f"\n{subheader(test_header)}")
        print(f"   {info_text(f'Question: \"{scenario.question}\"')}")

        mode_results = {}

        for mode in MODES:
            mode_label = mode_name(mode.key, f"{mode.icon} {mode.name}")
            print(f"\n  {mode_label}...")

            example_module = example_modules.get(mode.key)

            if example_module is None:
                skip_msg = status_skip("⏸️  SKIPPED (module not available)")
                print(f"  Result: {skip_msg}")
                mode_results[mode.key] = ModeResult(result="SKIPPED", found=False, matched=[], reply="")
                continue

            try:
                expect_memory = mode.success_metric != "forgot"

                # Handle different module interfaces
                if mode.key == "with_kusto":
                    passed, conversations, stats = example_module.run_example()
                else:
                    passed, conversations, stats = run_scenarios(
                        example_module.graph, [scenario], expect_memory=expect_memory
                    )

                conversation = conversations[0] if conversations else []
                print_conversation(conversation, scenario_num=None)

                match_info = stats["matches"][0] if stats.get("matches") else {}
                found = match_info.get("found", False)
                matched = match_info.get("matched", [])
                reply = match_info.get("reply", "")

                if mode.success_metric == "forgot":
                    if not found:
                        result = "FAIL"
                        result_msg = status_fail("❌ FAIL (forgot)")
                    else:
                        result = "UNEXPECTED"
                        result_msg = status_unexpected(f"⚠️  Unexpected recall (found: {matched})")
                else:
                    if found:
                        result = "PASS"
                        result_msg = status_pass(f"✅ PASS (remembered: {matched})")
                    else:
                        result = "FAIL"
                        result_msg = status_fail("❌ FAIL (forgot)")

                print(f"  Result: {result_msg}")
                mode_results[mode.key] = ModeResult(result=result, found=found, matched=matched, reply=reply)

            except Exception as e:
                skip_msg = status_skip(f"⏸️  SKIPPED ({str(e)[:50]})")
                print(f"  Result: {skip_msg}")
                mode_results[mode.key] = ModeResult(result="SKIPPED", found=False, matched=[], reply="")

        test_result = ScenarioResult(
            test_num=i,
            messages=scenario.messages,
            question=scenario.question,
            expected=scenario.expected_terms,
            without_memory=mode_results.get("without_memory", ModeResult("SKIPPED", False, [], "")),
            with_memory=mode_results.get("with_memory", ModeResult("SKIPPED", False, [], "")),
            with_kusto=mode_results.get("with_kusto", ModeResult("SKIPPED", False, [], "")),
        )
        results.append(test_result)

    print_summary(results)


def print_summary(results: list[ScenarioResult]):
    """Print summary of all test results."""
    print("\n" + banner_line("=" * 70))
    print(banner_line("📋 SUMMARY"))
    print(banner_line("=" * 70))

    total = len(results)

    for mode in MODES:
        mode_label = mode_name(mode.key, f"{mode.icon} {mode.name}")
        print(f"\n{mode_label}:")

        counts = {"PASS": 0, "FAIL": 0, "SKIPPED": 0, "UNEXPECTED": 0}
        for result in results:
            mode_result = getattr(result, mode.key)
            counts[mode_result.result] = counts.get(mode_result.result, 0) + 1

        if mode.success_metric == "forgot":
            print(f"   Expected: Forget after reset")
            print(f"   Result: {counts['FAIL']}/{total} forgot (as expected)")
            if counts["UNEXPECTED"] > 0:
                print(f"   ⚠️  Warning: {counts['UNEXPECTED']}/{total} unexpectedly remembered!")
        else:
            print(f"   Expected: Remember after reset")
            print(f"   Result: {counts['PASS']}/{total} remembered")

        if counts["SKIPPED"] > 0:
            print(f"   ⏸️  {counts['SKIPPED']}/{total} skipped")

    print("\n" + banner_line("=" * 70))


if __name__ == "__main__":
    run_all_examples()
