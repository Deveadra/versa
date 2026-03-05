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

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "exit_code": int(self.exit_code),
            "duration_ms": float(self.duration_ms),
            "stdout_tail": self.stdout_tail,
            "stderr_tail": self.stderr_tail,
            "parsed": self.parsed if isinstance(self.parsed, dict) else {},
        }


@dataclass(frozen=True)
class ScoreboardRun:
    mode: str  # "all" | "changed"
    fix_enabled: bool
    tool_results: dict[str, ToolResult]
    total_duration_ms: float

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "fix_enabled": bool(self.fix_enabled),
            "total_duration_ms": float(self.total_duration_ms),
            "tool_results": {
                name: (
                    tr.to_dict()
                    if hasattr(tr, "to_dict")
                    else {
                        "name": getattr(tr, "name", name),
                        "exit_code": int(getattr(tr, "exit_code", 1)),
                        "duration_ms": float(getattr(tr, "duration_ms", 0.0)),
                        "stdout_tail": getattr(tr, "stdout_tail", ""),
                        "stderr_tail": getattr(tr, "stderr_tail", ""),
                        "parsed": getattr(tr, "parsed", {}) or {},
                    }
                )
                for name, tr in self.tool_results.items()
            },
            # gates_failing is a COUNT (int), not an iterable
            "gates_failing": int(self.gates_failing),
            # Optional but very useful for debugging/reporting
            "failing_tools": [
                name
                for name, tr in self.tool_results.items()
                if int(getattr(tr, "exit_code", 1)) != 0
            ],
            "score": float(self.score()),
            "passed": bool(self.passed()),
        }

    @property
    def gates_failing(self) -> int:
        return sum(1 for tr in self.tool_results.values() if tr.exit_code != 0)

    def passed(self) -> bool:
        return self.gates_failing == 0

    def score(self) -> float:
        ruff_count = int(
            self.tool_results.get(
                "ruff", ToolResult("ruff", 0, 0, "", "", {"count": 0})
            ).parsed.get("count", 0)
        )
        pytest_failures = int(
            self.tool_results.get(
                "pytest", ToolResult("pytest", 0, 0, "", "", {"failures": 0})
            ).parsed.get("failures", 0)
        )
        compile_failures = int(
            self.tool_results.get(
                "compile", ToolResult("compile", 0, 0, "", "", {"failures": 0})
            ).parsed.get("failures", 0)
        )
        total_sec = self.total_duration_ms / 1000.0

        return (
            1000.0
            - 300.0 * self.gates_failing
            - 1.0 * ruff_count
            - 50.0 * pytest_failures
            - 10.0 * compile_failures
            - 0.5 * total_sec
        )
