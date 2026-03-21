from __future__ import annotations

import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class TestResult:
    ok: bool
    cmd: list[str]
    rc: int
    stdout: str
    stderr: str


class TestRunner:
    def __init__(self, repo_root: str | Path) -> None:
        self.repo_root = str(Path(repo_root).resolve())

    def _run(self, args: Sequence[str], timeout_sec: int = 600) -> TestResult:
        cmd = list(args)
        completed = subprocess.run(
            args=cmd,
            cwd=self.repo_root,
            text=True,
            capture_output=True,
            timeout=timeout_sec,
            check=False,
        )
        return TestResult(
            ok=(completed.returncode == 0),
            cmd=cmd,
            rc=completed.returncode,
            stdout=completed.stdout[-20000:],
            stderr=completed.stderr[-20000:],
        )

    def ruff(self, paths: Sequence[str] | None = None) -> TestResult:
        cmd = [sys.executable, "-m", "ruff", "check"]
        cmd.extend(paths or ["."])
        return self._run(cmd)

    def ruff_fix(self, paths: Sequence[str] | None = None) -> TestResult:
        cmd = [sys.executable, "-m", "ruff", "check", "--fix"]
        cmd.extend(paths or ["."])
        return self._run(cmd)

    def black_check(self, paths: Sequence[str] | None = None) -> TestResult:
        cmd = [sys.executable, "-m", "black", "--check"]
        cmd.extend(paths or ["."])
        return self._run(cmd)

    def black_format(self, paths: Sequence[str] | None = None) -> TestResult:
        cmd = [sys.executable, "-m", "black"]
        cmd.extend(paths or ["."])
        return self._run(cmd)

    def pytest_quick(self, paths: Sequence[str] | None = None) -> TestResult:
        cmd = [sys.executable, "-m", "pytest", "-q"]
        if paths:
            cmd.extend(paths)
        return self._run(cmd)
