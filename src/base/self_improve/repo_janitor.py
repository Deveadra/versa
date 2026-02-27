from __future__ import annotations

from dataclasses import dataclass
from typing import Any

@dataclass
class JanitorFinding:
    kind: str          # "lint" | "format" | "test" | "security" | "dead_code"
    path: str
    detail: str
    autofixable: bool
    severity: int      # 1-10

class RepoJanitor:
    def __init__(self, repo_root: str, test_runner, patcher) -> None:
        self.repo_root = repo_root
        self.tests = test_runner
        self.patcher = patcher

    def scan(self) -> list[JanitorFinding]:
        # v0: only use tool outputs; avoid bespoke static analysis until later
        findings: list[JanitorFinding] = []
        ruff = self.tests.ruff()
        if not ruff.ok:
            findings.append(JanitorFinding("lint", ".", "ruff failing", True, 6))
        black = self.tests.black_check()
        if not black.ok:
            findings.append(JanitorFinding("format", ".", "black --check failing", True, 5))
        return findings

    def propose_autofix(self, findings: list[JanitorFinding]) -> dict[str, Any]:
        # v0: map known issues to known fixes (e.g., ruff --fix, black)
        return {"actions": ["ruff_fix", "black_format"], "risk": "low"}
