from __future__ import annotations

from collections.abc import Sequence
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from defusedxml import ElementTree as ET
from loguru import logger

from base.self_improve.score_types import ScoreboardRun, ToolResult

_ALLOWED_TOOLS = {"ruff", "black", "compileall"}


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

def _validate_internal_cmd(cmd: Sequence[str]) -> list[str]:
    """
    Validate an argv-style command that is constructed internally (not user-provided).

    Keeps security scanners happy and prevents accidental misuse:
    - must be a non-empty list/tuple of strings
    - no NUL bytes
    """
    if not isinstance(cmd, (list, tuple)) or not cmd:
        raise TypeError(f"cmd must be a non-empty list/tuple of strings, got: {type(cmd)!r}")

    out: list[str] = []
    for part in cmd:
        if not isinstance(part, str):
            raise TypeError(f"cmd contains non-string element: {part!r}")
        if "\x00" in part:
            raise ValueError("cmd contains NUL byte")
        out.append(part)

    return out

def _run_tool(tool: str, cwd: Path, timeout_sec: int = 600) -> tuple[int, str, str, float]:
    """
    Run only a whitelisted tool with a fixed argv structure.
    This prevents arbitrary command execution and keeps scanners satisfied.
    """
    if tool not in _ALLOWED_TOOLS:
        raise ValueError(f"refused to run unknown tool: {tool}")

    if tool == "ruff":
        cmd = [sys.executable, "-m", "ruff", "check", "--output-format=json", "."]
    elif tool == "black":
        cmd = [sys.executable, "-m", "black", "--check", "."]
    else:  # compileall
        cmd = [sys.executable, "-m", "compileall", "-q", "."]

    argv = _validate_internal_cmd(cmd)
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"

    start = time.perf_counter()
    # nosemgrep: python.lang.security.audit.dangerous-subprocess-use-audit
    # Justification: argv is constructed internally from a strict allowlist (shell=False) and validated above.
    proc = subprocess.run(
        argv,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout_sec,
        env=env,
    )
    dur_ms = (time.perf_counter() - start) * 1000.0
    return proc.returncode, proc.stdout or "", proc.stderr or "", dur_ms


def _run_pytest_with_junit(
    cwd: Path, junit_path: Path, timeout_sec: int = 1200
) -> tuple[int, str, str, float]:
    """
    Pytest is run via a fixed argv structure.
    junit_path is a Path created by our code (not user-supplied).
    """
    cmd = [sys.executable, "-m", "pytest", "-q", f"--junitxml={junit_path}"]

    argv = _validate_internal_cmd(cmd)
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"

    start = time.perf_counter()
    # nosemgrep: python.lang.security.audit.dangerous-subprocess-use-audit
    # Justification: argv is constructed internally (shell=False) and validated above; junit_path is internal.
    proc = subprocess.run(
        argv,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout_sec,
        env=env,
    )
    dur_ms = (time.perf_counter() - start) * 1000.0
    return proc.returncode, proc.stdout or "", proc.stderr or "", dur_ms


def _parse_pytest_junit(junit_path: Path) -> dict[str, Any]:
    if not junit_path.exists():
        return {"failures": 0, "errors": 0, "tests": 0, "skipped": 0}

    tree = ET.parse(str(junit_path))
    root = tree.getroot()

    if root is None:
        return {"failures": 0, "errors": 0, "tests": 0, "skipped": 0}

    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    # suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))

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

        # Ruff
        rc, out, err, ms = _run_tool("ruff", self.repo, timeout_sec=600)
        try:
            parsed = {"count": len(json.loads(out))} if out.strip() else {"count": 0}
        except Exception:
            parsed = {"count": 0, "parse_error": True}
        results["ruff"] = ToolResult("ruff", rc, ms, _tail(out), _tail(err), parsed)

        # Black
        rc, out, err, ms = _run_tool("black", self.repo, timeout_sec=600)
        results["black"] = ToolResult("black", rc, ms, _tail(out), _tail(err), {})

        # Pytest (junit)
        reports_dir = self.repo / "reports" / "pytest"
        reports_dir.mkdir(parents=True, exist_ok=True)
        junit_path = reports_dir / "junit.xml"

        rc, out, err, ms = _run_pytest_with_junit(self.repo, junit_path, timeout_sec=1200)
        parsed = _parse_pytest_junit(junit_path)
        results["pytest"] = ToolResult("pytest", rc, ms, _tail(out), _tail(err), parsed)

        # Syntax compile
        rc, out, err, ms = _run_tool("compileall", self.repo, timeout_sec=600)
        parsed = {"failures": 0 if rc == 0 else 1}
        results["compile"] = ToolResult("compile", rc, ms, _tail(out), _tail(err), parsed)

        total_ms = (time.perf_counter() - t0) * 1000.0
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