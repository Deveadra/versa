from __future__ import annotations

import shlex
import subprocess
import sys
from dataclasses import dataclass


@dataclass
class TestResult:
    ok: bool
    cmd: list[str]
    rc: int
    stdout: str
    stderr: str


class TestRunner:
    def __init__(self, repo_root: str) -> None:
        self.repo_root = repo_root

    def _run(self, cmd: list[str], timeout_sec: int = 600) -> TestResult:
        p = subprocess.run(
            cmd=[shlex.quote(arg) for arg in cmd],
            cwd=self.repo_root,
            text=True,
            capture_output=True,
            timeout=timeout_sec,
            check=False,
        )
        return TestResult(
            ok=(p.returncode == 0),
            cmd=cmd,
            rc=p.returncode,
            stdout=p.stdout[-20000:],
            stderr=p.stderr[-20000:],
        )

    def ruff(self) -> TestResult:
        return self._run([sys.executable, "-m", "ruff", "check", "."])

    def black_check(self) -> TestResult:
        return self._run([sys.executable, "-m", "black", "--check", "."])

    def pytest_quick(self) -> TestResult:
        return self._run([sys.executable, "-m", "pytest", "-q"])
