# src/base/self_improve/score_types.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolResult:
    name: str
    exit_code: int
    duration_ms: float
    stdout_tail: str
    stderr_tail: str
    parsed: dict[str, Any]


@dataclass(frozen=True)
class ScoreboardRun:
    mode: str  # "all" | "changed"
    fix_enabled: bool
    tool_results: dict[str, ToolResult]
    total_duration_ms: float

    @property
    def gates_failing(self) -> int:
        return sum(1 for tr in self.tool_results.values() if tr.exit_code != 0)

    def passed(self) -> bool:
        return self.gates_failing == 0

    def score(self) -> float:
        ruff_count = int(self.tool_results.get("ruff", ToolResult("ruff", 0, 0, "", "", {"count": 0})).parsed.get("count", 0))
        pytest_failures = int(self.tool_results.get("pytest", ToolResult("pytest", 0, 0, "", "", {"failures": 0})).parsed.get("failures", 0))
        compile_failures = int(self.tool_results.get("compile", ToolResult("compile", 0, 0, "", "", {"failures": 0})).parsed.get("failures", 0))
        total_sec = self.total_duration_ms / 1000.0

        return (
            1000.0
            - 300.0 * self.gates_failing
            - 1.0 * ruff_count
            - 50.0 * pytest_failures
            - 10.0 * compile_failures
            - 0.5 * total_sec
        )
