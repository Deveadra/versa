from __future__ import annotations

from base.self_improve.score_types import ScoreboardRun, ToolResult


def test_score_decreases_with_failures():
    run_ok = ScoreboardRun(
        mode="all",
        fix_enabled=False,
        tool_results={
            "ruff": ToolResult("ruff", 0, 10, "", "", {"count": 0}),
            "black": ToolResult("black", 0, 10, "", "", {}),
            "pytest": ToolResult("pytest", 0, 50, "", "", {"failures": 0}),
            "compile": ToolResult("compile", 0, 10, "", "", {"failures": 0}),
        },
        total_duration_ms=1000,
    )
    run_bad = ScoreboardRun(
        mode="all",
        fix_enabled=False,
        tool_results={
            "ruff": ToolResult("ruff", 1, 10, "", "", {"count": 10}),
            "black": ToolResult("black", 1, 10, "", "", {}),
            "pytest": ToolResult("pytest", 1, 50, "", "", {"failures": 2}),
            "compile": ToolResult("compile", 0, 10, "", "", {"failures": 0}),
        },
        total_duration_ms=1000,
    )
    assert run_ok.score() > run_bad.score()
