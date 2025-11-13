from examples.models import ModeConfig, TestCase

TEST_CASES = [
    TestCase(
        messages=["I like apples", "Honey Crisp"],
        question="What fruit do I like?",
        expected_terms=["apple", "honeycrisp"]
    ),
    TestCase(
        messages=["I love strawberries", "the organic ones"],
        question="What fruit do I love?",
        expected_terms=["strawberry", "strawberries", "organic"]
    ),
    TestCase(
        messages=["My favorite color is blue", "Navy blue specifically"],
        question="What is my favorite color?",
        expected_terms=["blue", "navy"]
    ),
]

MODES = [
    ModeConfig(
        key="without_memory",
        name="WITHOUT Memory (Thread-based only)",
        icon="1)",
        expected="Forget after reset (true negative)",
        success_metric="forgot",
        has_unexpected=True
    ),
    ModeConfig(
        key="with_memory",
        name="WITH Memory (InMemoryStore)",
        icon="2)",
        expected="Remember after reset (true positive)",
        success_metric="passed",
        has_unexpected=False
    ),
    ModeConfig(
        key="with_kusto",
        name="WITH Memory (Kusto)",
        icon="3)",
        expected="Remember after reset (true positive)",
        success_metric="passed",
        has_unexpected=False
    )
]
