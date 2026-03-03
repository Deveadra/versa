from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from defusedxml.etree import ElementTree as ET
from dataclasses import dataclass
from loguru import logger
from pathlib import Path
from typing import Any

from base.self_improve.score_types import ScoreboardRun, ToolResult



_ALLOWED_TOOLS = {"ruff", "black", "pytest", "compileall"}


@dataclass
class ScoreboardSnapshot:
    ts: str
    dream_runs_total: int
    dream_runs_passed: int
    prs_opened_total: int
    last_run_status: str


def _tail(text: str, max_lines: int = 80, max_chars: int = 8000) -> str:
    lines = (text or "").splitlines()[-max_lines:]
    out = "\n".join(lines)
    return out[-max_chars:]

def _run_tool(tool: str, cwd: Path, timeout_sec: int = 600) -> tuple[int, str, str, float]:
    if tool not in _ALLOWED_TOOLS:
        raise ValueError(f"refused to run unknown tool: {tool}")

    if tool == "ruff":
        cmd = [sys.executable, "-m", "ruff", "check", "--output-format=json", "."]
    elif tool == "black":
        cmd = [sys.executable, "-m", "black", "--check", "."]
    elif tool == "pytest":
    raise RuntimeError("pytest requires junit_path; call _run_pytest_with_junit(...)")
        # caller creates junit_path; this function runs tests only
        cmd = [sys.executable, "-m", "pytest", "-q"]
    else:  # compileall
        cmd = [sys.executable, "-m", "compileall", "-q", "."]

    start = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout_sec,
        env={**os.environ, "PYTHONUTF8": "1"},
    )
    dur_ms = (time.perf_counter() - start) * 1000
    return proc.returncode, proc.stdout or "", proc.stderr or "", dur_ms

def _run_pytest_with_junit(cwd: Path, junit_path: Path, timeout_sec: int = 1200) -> tuple[int, str, str, float]:
    cmd = [sys.executable, "-m", "pytest", "-q", f"--junitxml={junit_path}"]
    start = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout_sec,
        env={**os.environ, "PYTHONUTF8": "1"},
    )
    dur_ms = (time.perf_counter() - start) * 1000
    return proc.returncode, proc.stdout or "", proc.stderr or "", dur_ms

def _parse_pytest_junit(junit_path: Path) -> dict[str, Any]:
    if not junit_path.exists():
        return {"failures": 0, "errors": 0, "tests": 0, "skipped": 0}

    tree = ET.parse(str(junit_path))
    root = tree.getroot()

    # Sometimes root is <testsuite>, sometimes <testsuites>. Aggregate.
    suites = []
    if root.tag == "testsuite":
        suites = [root]
    else:
        suites = list(root.findall("testsuite"))

    def _sum(attr: str) -> int:
        total = 0
        for s in suites:
            try:
                total += int(s.attrib.get(attr, "0"))
            except Exception:
                pass
        return total

    return {
        "tests": _sum("tests"),
        "failures": _sum("failures"),
        "errors": _sum("errors"),
        "skipped": _sum("skipped"),
    }


class ScoreboardRunner:
    def __init__(self, repo_root: str | Path):
        self.repo = Path(repo_root).resolve()

    def run(self, *, mode: str = "all", fix: bool = False) -> ScoreboardRun:
        t0 = time.perf_counter()
        results: dict[str, ToolResult] = {}

        # Ruff: JSON for machine-readability
        ruff_cmd = [sys.executable, "-m", "ruff", "check", "--output-format=json", "."]
        rc, out, err, ms = _run_tool("ruff", self.repo, timeout_sec=600)
        parsed = {}
        try:
            parsed = {"count": len(json.loads(out))} if out.strip() else {"count": 0}
        except Exception:
            parsed = {"count": 0, "parse_error": True}
        results["ruff"] = ToolResult("ruff", rc, ms, _tail(out), _tail(err), parsed)

        # Black: deterministic exit codes; no parsing required
        black_cmd = [sys.executable, "-m", "black", "--check", "."]
        rc, out, err, ms = _run(black_cmd, self.repo, timeout_sec=600)
        results["black"] = ToolResult("black", rc, ms, _tail(out), _tail(err), {})

        # Pytest: write junit xml for parseable totals
        reports_dir = self.repo / "reports" / "pytest"
        reports_dir.mkdir(parents=True, exist_ok=True)
        junit_path = reports_dir / "junit.xml"
        pytest_cmd = [sys.executable, "-m", "pytest", "-q", f"--junitxml={junit_path}"]
        rc, out, err, ms = _run(pytest_cmd, self.repo, timeout_sec=1200)
        parsed = _parse_pytest_junit(junit_path)
        results["pytest"] = ToolResult("pytest", rc, ms, _tail(out), _tail(err), parsed)

        # Syntax compile
        comp_cmd = [sys.executable, "-m", "compileall", "-q", "."]
        rc, out, err, ms = _run(comp_cmd, self.repo, timeout_sec=600)
        parsed = {"failures": 0 if rc == 0 else 1}
        results["compile"] = ToolResult("compile", rc, ms, _tail(out), _tail(err), parsed)

        total_ms = (time.perf_counter() - t0) * 1000
        run = ScoreboardRun(mode=mode, fix_enabled=fix, tool_results=results, total_duration_ms=total_ms)
        logger.info(f"[scoreboard] gates_failing={run.gates_failing} score={run.score():.2f}")
        return run
    
    
class Scoreboard:
    def __init__(self, conn) -> None:
        self.conn = conn

    def bump(self, key: str, delta: int = 1) -> None:
        self.conn.execute(
            "INSERT INTO scoreboard(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=value+?",
            (key, delta, delta),
        )
        self.conn.commit()
