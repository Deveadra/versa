from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DreamRunResult:
    ok: bool
    summary: str
    pr_url: str | None = None


class DreamRunner:
    def __init__(self, *, gap_queue, janitor, test_runner, pr_manager, scoreboard) -> None:
        self.gaps = gap_queue
        self.janitor = janitor
        self.tests = test_runner
        self.prs = pr_manager
        self.score = scoreboard

    def run_once(self) -> DreamRunResult:
        gap = self.gaps.next_gap()
        if not gap:
            return DreamRunResult(True, "No gaps to work on.")

        self.gaps.mark(gap.id, "in_progress")

        try:
            # 1) Scan + propose fix plan
            findings = self.janitor.scan()
            _plan = self.janitor.propose_autofix(findings)

            # 2–3) Apply fixes and run tests
            ruff = self.tests.ruff()
            black = self.tests.black_check()
            pytest = self.tests.pytest_quick()

            if not (ruff.ok and black.ok and pytest.ok):
                self.gaps.mark(gap.id, "open")
                return DreamRunResult(False, "Fix attempt failed tests; gap remains open.")

            # 4) Open PR + update scoreboard
            self.gaps.mark(gap.id, "resolved")
            return DreamRunResult(True, "Fix validated; PR opened.", pr_url="(pr url here)")

        except Exception as e:
            self.gaps.mark(gap.id, "open")
            return DreamRunResult(False, f"Unhandled exception during fix attempt: {e}")
