from examples.test_data import TEST_CASES, MODES
from examples.models import TestResult, ModeResult, Statistics, ModeStatistics
from examples.utils import print_conversation
from core.runner import run_tests
from examples.styling import (
    banner_line, header, subheader, mode_name,
    status_pass, status_fail, status_skip, status_unexpected,
    table_header, table_separator, info_text
)

# Explicit imports for example modules
import examples.without_memory as without_memory_module
import examples.with_memory as with_memory_module

def run_all_examples():
    """Run all memory examples and compare results"""
    print("\n" + banner_line("="*70))
    print(banner_line("🤖 LangGraph ReAct Memory Demo - LM Studio"))
    print(banner_line("="*70))
    print(f"\n{subheader('📊 Running')} {len(TEST_CASES)} {subheader('test case(s)...')}")
    
    # Map mode keys to imported modules
    example_modules = {
        "without_memory": without_memory_module,
        "with_memory": with_memory_module,
        "with_kusto": None  # Not yet implemented
    }
    
    results = []
    
    # Run each test case
    for i, test_case in enumerate(TEST_CASES, 1):
        if i > 1:
            # Add separator between test cases
            print("\n" + table_separator("=" * 70))
        test_header = f"📝 Test #{i}: {' → '.join(test_case.messages)}"
        print(f"\n{subheader(test_header)}")
        print(f"   {info_text(f'Question: \"{test_case.question}\"')}")
        
        mode_results = {}
        
        # Run each mode
        for mode in MODES:
            mode_label = mode_name(mode.key, f"{mode.icon} {mode.name}")
            print(f"\n  {mode_label}...")
            
            example_module = example_modules.get(mode.key)
            
            if example_module is None:
                skip_msg = status_skip("⏸️  SKIPPED (module not available)")
                print(f"  Result: {skip_msg}")
                mode_results[mode.key] = ModeResult(
                    result="SKIPPED",
                    found=False,
                    matched=[],
                    reply=""
                )
                continue
            
            try:
                # Use the runner helper instead of run_example
                expect_memory = mode.success_metric != "forgot"
                passed, conversations, stats = run_tests(
                    example_module.graph,
                    [test_case],
                    expect_memory=expect_memory
                )
                
                # Get the conversation for this test
                conversation = conversations[0] if conversations else []
                print_conversation(conversation)
                
                # Get the match info
                match_info = stats["matches"][0] if stats.get("matches") else {}
                found = match_info.get("found", False)
                matched = match_info.get("matched", [])
                reply = match_info.get("reply", "")
                
                # Determine result
                if mode.success_metric == "forgot":
                    # Without memory mode - should forget
                    if not found:
                        result = "FAIL"  # Correctly forgot
                        result_msg = status_fail("❌ FAIL (forgot)")
                        print(f"  Result: {result_msg}")
                    else:
                        result = "UNEXPECTED"  # Unexpectedly remembered
                        result_msg = status_unexpected(f"⚠️  Unexpected recall (found: {matched})")
                        print(f"  Result: {result_msg}")
                else:
                    # Memory modes - should remember
                    if found:
                        result = "PASS"
                        result_msg = status_pass(f"✅ PASS (remembered: {matched})")
                        print(f"  Result: {result_msg}")
                    else:
                        result = "FAIL"
                        result_msg = status_fail("❌ FAIL (forgot)")
                        print(f"  Result: {result_msg}")
                
                mode_results[mode.key] = ModeResult(
                    result=result,
                    found=found,
                    matched=matched,
                    reply=reply
                )
                
            except NotImplementedError as e:
                skip_msg = status_skip(f"⏸️  SKIPPED ({str(e)})")
                print(f"  Result: {skip_msg}")
                mode_results[mode.key] = ModeResult(
                    result="SKIPPED",
                    found=False,
                    matched=[],
                    reply=""
                )
        
        # Create test result
        test_result = TestResult(
            test_num=i,
            messages=test_case.messages,
            question=test_case.question,
            expected=test_case.expected_terms,
            without_memory=mode_results.get("without_memory", ModeResult("SKIPPED", False, [], "")),
            with_memory=mode_results.get("with_memory", ModeResult("SKIPPED", False, [], "")),
            with_kusto=mode_results.get("with_kusto", ModeResult("SKIPPED", False, [], ""))
        )
        results.append(test_result)
    
    # Print summary
    print("\n" + banner_line("="*70))
    print(banner_line("📋 SUMMARY"))
    print(banner_line("="*70))
    
    stats = calculate_statistics(results)
    print_statistics_summary(stats)
    print_detailed_results_table(results)
    print_overall_verdict(stats)

def calculate_statistics(results: list[TestResult]) -> Statistics:
    """Calculate test statistics from results"""
    stats = Statistics(total=len(results))
    
    for mode in MODES:
        mode_stats = ModeStatistics()
        
        for result in results:
            mode_result = getattr(result, mode.key)
            result_value = mode_result.result
            mode_stats.counts[result_value] = mode_stats.counts.get(result_value, 0) + 1
        
        stats.modes[mode.key] = mode_stats
    
    return stats

def format_status(result: str, mode_key: str) -> str:
    """Format status indicator for a given result"""
    if mode_key == "without_memory":
        status_map = {
            "FAIL": "❌ Forgot",
            "UNEXPECTED": "⚠️  Remembered",
        }
        return status_map.get(result, "? Unknown")
    
    status_map = {
        "PASS": "✅ PASS",
        "FAIL": "❌ FAIL",
        "SKIPPED": "⏸️  SKIP",
    }
    return status_map.get(result, "? Unknown")

def print_statistics_summary(stats: Statistics):
    """Print statistical summary of test results"""
    total = stats.total
    
    for mode in MODES:
        mode_key = mode.key
        mode_stats = stats.modes[mode_key]
        
        mode_label = mode_name(mode.key, f"{mode.icon} {mode.name}")
        print(f"\n{mode_label}:")
        print(f"   {info_text(f'Expected: {mode.expected}')}")
        
        if mode.success_metric == "forgot":
            success_count = mode_stats.counts.get("FAIL", 0)
            print(f"   Result: {success_count}/{total} forgot (as expected)")
        else:
            print(f"   Result: {mode_stats.counts.get('PASS', 0)}/{total} tests passed")
        
        if mode.has_unexpected and mode_stats.counts.get("UNEXPECTED", 0) > 0:
            print(f"   ⚠️  Warning: {mode_stats.counts['UNEXPECTED']}/{total} unexpectedly remembered!")
        
        if mode_stats.counts.get("SKIPPED", 0) > 0:
            print(f"   Result: {mode_stats.counts['SKIPPED']}/{total} tests skipped (not implemented)")

def print_detailed_results_table(results: list[TestResult]):
    """Print detailed results in table format"""
    headers = ["Test"] + [mode.name.split("(")[0].strip() for mode in MODES] + ["Messages"]
    col_widths = [6, 18, 13, 13, 25]
    
    print()
    for i, header in enumerate(headers):
        styled_header = table_header(header)
        # Note: termcolor handles padding, so just print with spacing
        print(f"{header:<{col_widths[i]}}", end=" ")
    print()
    print(table_separator("-" * sum(col_widths)))
    
    for r in results:
        print(f"#{r.test_num:<{col_widths[0]-1}}", end=" ")
        
        for i, mode in enumerate(MODES):
            mode_result = getattr(r, mode.key)
            status = format_status(mode_result.result, mode.key)
            print(f"{status:<{col_widths[i+1]}}", end=" ")
        
        msgs = ' → '.join(r.messages[:2])
        if len(msgs) > col_widths[-1]:
            msgs = msgs[:col_widths[-1]-3] + "..."
        print(msgs)

def print_overall_verdict(stats: Statistics):
    """Print overall test verdict"""
    total = stats.total
    
    print("\n" + banner_line("="*80))
    
    all_passed = True
    failure_messages = []
    
    for mode in MODES:
        mode_key = mode.key
        mode_stats = stats.modes[mode_key]
        
        if mode.success_metric == "forgot":
            expected_count = mode_stats.counts.get("FAIL", 0)
            unexpected_count = mode_stats.counts.get("UNEXPECTED", 0)
            
            if expected_count != total:
                all_passed = False
            
            if unexpected_count > 0:
                all_passed = False
                failure_messages.append(f"   - {mode.name}: {unexpected_count} tests unexpectedly remembered")
        else:
            passed_count = mode_stats.counts.get("PASS", 0)
            skipped_count = mode_stats.counts.get("SKIPPED", 0)
            
            if passed_count < total and skipped_count == 0:
                all_passed = False
                failure_messages.append(f"   - {mode.name}: {total - passed_count} tests failed to remember")
    
    if all_passed:
        success_msg = status_pass("🎉 MEMORY SYSTEM WORKING AS EXPECTED!")
        print(success_msg)
        for mode in MODES:
            mode_key = mode.key
            mode_stats = stats.modes[mode_key]
            
            if mode.success_metric == "forgot":
                count = mode_stats.counts.get("FAIL", 0)
                mode_label = mode_name(mode.key, mode.name)
                print(f"   - {mode_label}: All {count} tests forgot (expected behavior)")
            else:
                passed = mode_stats.counts.get("PASS", 0)
                skipped = mode_stats.counts.get("SKIPPED", 0)
                
                mode_label = mode_name(mode.key, mode.name)
                if skipped > 0:
                    skip_info = info_text(f"{skipped} tests skipped (not yet implemented)")
                    print(f"   - {mode_label}: {skip_info}")
                elif passed == total:
                    print(f"   - {mode_label}: All {passed} tests remembered (desired behavior)")
    else:
        warning_msg = status_unexpected("⚠️  UNEXPECTED RESULTS DETECTED")
        print(warning_msg)
        for msg in failure_messages:
            print(msg)
    
    print(banner_line("="*80) + "\n")

if __name__ == "__main__":
    run_all_examples()
