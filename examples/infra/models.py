from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(slots=True)
class ModeConfig:
    """Configuration for a memory mode"""

    key: str
    name: str
    icon: str
    expected: str
    success_metric: str  # "forgot" or "passed"
    has_unexpected: bool = False


@dataclass(slots=True)
class ModeResult:
    """Result for a single mode in a test"""

    result: str  # "PASS", "FAIL", "UNEXPECTED", "SKIPPED"
    found: bool
    matched: list[str]
    reply: str


@dataclass(slots=True)
class ScenarioResult:
    """Result for a single test case across all modes"""

    test_num: int
    messages: list[str]
    question: str
    expected: list[str]
    without_memory: ModeResult
    with_memory: ModeResult
    with_kusto: ModeResult
    store: Optional[Any] = None
    store_kusto: Optional[Any] = None


@dataclass(slots=True)
class ModeStatistics:
    """Statistics for a single mode"""

    counts: dict[str, int] = field(default_factory=dict)

    def get(self, key: str, default: int = 0) -> int:
        return self.counts.get(key, default)


@dataclass(slots=True)
class Statistics:
    """Overall test statistics"""

    total: int
    modes: dict[str, ModeStatistics] = field(default_factory=dict)

    def __getitem__(self, key: str):
        if key == "total":
            return self.total
        return self.modes.get(key, ModeStatistics())


@dataclass(slots=True)
class Scenario:
    """Single test case definition"""

    messages: list[str]
    question: str
    expected_terms: list[str]
