from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from base.database.sqlite import SQLiteConn
from base.self_improve.iteration_controller import (
    RepoJanitorIterationController as IterationController,
)
from config.config import settings


class FakeScoreboardResult:
    def __init__(self, *, gates_failing: int, score_value: float, comparison_score: float = 1000.0):
        self.gates_failing = int(gates_failing)
        self._score_value = float(score_value)
        self.comparison_score = float(comparison_score)

        # These get filled by the scoreboard stub
        self.artifact_path: str = ""
        self.artifact_relpath: str = ""

    def score(self) -> float:
        return float(self._score_value)

    def passed(self) -> bool:
        return self.gates_failing == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "gates_failing": self.gates_failing,
            "score": float(self._score_value),
            "comparison_score": float(self.comparison_score),
        }


class ScoreboardStub:
    def __init__(self, *, repo: Path, results: list[FakeScoreboardResult]):
        self.repo = repo
        self._results = list(results)
        self.calls: list[dict[str, Any]] = []

    def run(self, *, mode: str, fix: bool, artifact_path: Path, context: dict[str, Any]):
        self.calls.append(
            {
                "mode": mode,
                "fix": bool(fix),
                "artifact_path": str(artifact_path),
                "context": context,
            }
        )

        assert self._results, "ScoreboardStub: no more results queued"
        r = self._results.pop(0)

        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(json.dumps(r.to_dict(), ensure_ascii=False), encoding="utf-8")

        r.artifact_path = str(artifact_path)
        try:
            r.artifact_relpath = artifact_path.relative_to(self.repo).as_posix()
        except Exception:
            r.artifact_relpath = str(artifact_path)

        return r


class PRManagerStub:
    def prepare_branch(self, branch_name: str, base: str | None = None) -> str:
        return branch_name


def _mk_controller(
    tmp_path: Path, db: SQLiteConn, scoreboard: ScoreboardStub
) -> IterationController:
    ctl: IterationController = IterationController.__new__(IterationController)

    # minimal fields used by run()
    ctl.repo = tmp_path
    ctl.conn = db.conn
    ctl.scoreboard = scoreboard
    ctl.pr_manager = PRManagerStub()

    # ---- Hard stubs to avoid real git/IO side-effects ----
    ctl._git = lambda *a, **k: 0
    ctl._current_branch = lambda: "feature/flywheel"
    ctl._current_sha = lambda: "deadbeefdeadbeef"
    ctl._worktree_dirty = lambda: False
    ctl._rollback_to_base = lambda *a, **k: None

    # No-op status hooks
    ctl._status_start = lambda *a, **k: None
    ctl._status_finish = lambda *a, **k: None
    ctl._emit_status = lambda *a, **k: None

    # Avoid gap machinery (not the subject of these tests)
    ctl._goal_from_gaps = lambda limit=5: ("goal from gaps", [])
    ctl._log_gaps_from_scoreboard = lambda *a, **k: set()
    # Keep reconcile calls harmless (tables exist; no fingerprints returned)

    # Helpers used during run()
    ctl._artifact_relpath = lambda p: Path(p).relative_to(tmp_path).as_posix()

    def _write_json(path: Path, payload: dict[str, Any]):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    ctl._write_json = _write_json

    ctl._write_attempt_summary_artifact = lambda *, run_artifact_dir, iteration, branch, stage, improved, error, attempt_artifacts, extra=None: {
        "path": str(
            (run_artifact_dir / f"iteration_{iteration:02d}_attempt_summary.json").resolve()
        ),
        "relative_path": (run_artifact_dir / f"iteration_{iteration:02d}_attempt_summary.json")
        .relative_to(tmp_path)
        .as_posix(),
    }

    # Score comparisons: prefer comparison_score if present (matches your real behavior)
    ctl._score_value = lambda res: float(getattr(res, "comparison_score", res.score()))

    def _score_delta_summary(
        before: FakeScoreboardResult, after: FakeScoreboardResult
    ) -> dict[str, Any]:
        b = ctl._score_value(before)
        a = ctl._score_value(after)
        return {
            "before_score": b,
            "after_score": a,
            "score_delta": a - b,
            "before_gates": int(before.gates_failing),
            "after_gates": int(after.gates_failing),
            "gates_delta": int(after.gates_failing) - int(before.gates_failing),
            "newly_failing_tools": [],
            "resolved_tools": [],
            "still_failing_tools": [],
        }

    ctl._score_delta_summary = _score_delta_summary

    return ctl


@pytest.fixture
def db(tmp_path: Path) -> SQLiteConn:
    # SQLiteConn applies migrations on init in your project
    return SQLiteConn(str(tmp_path / "semantics.db"))


def _latest_attempt_row(db: SQLiteConn):
    row = db.conn.execute("""
        SELECT proposal_json, error_text
        FROM repo_improvement_attempts
        ORDER BY id DESC
        LIMIT 1
        """).fetchone()
    assert row is not None, "Expected at least one repo_improvement_attempts row"
    return row[0], row[1]


def test_safe_autofix_transient_improvement_is_not_error(
    tmp_path: Path, db: SQLiteConn, monkeypatch
):
    # Force the safe-autofix-only path.
    monkeypatch.setattr(settings, "self_improve_safe_autofix_only", True, raising=False)
    monkeypatch.setattr(
        settings, "self_improve_enable_llm_autonomous_changes", False, raising=False
    )
    monkeypatch.setattr(settings, "github_default_branch", "feature/flywheel", raising=False)

    # Health improves because gates drop (baseline=1 -> after=0), but worktree stays clean => transient.
    sb = ScoreboardStub(
        repo=tmp_path,
        results=[
            FakeScoreboardResult(gates_failing=1, score_value=10.0),  # baseline
            FakeScoreboardResult(gates_failing=1, score_value=10.0),  # before
            FakeScoreboardResult(gates_failing=0, score_value=10.0),  # after (fix=True)
            FakeScoreboardResult(gates_failing=0, score_value=10.0),  # final
        ],
    )

    ctl = _mk_controller(tmp_path, db, sb)

    budget = SimpleNamespace(
        max_iterations=1,
        max_seconds=999,
        gap_limit=5,
        open_pr_on_improvement=False,
        stop_on_first_improvement=False,
    )

    ctl.run(goal="test", budget=budget, status_callback=None)

    proposal_json_raw, error_text = _latest_attempt_row(db)
    assert error_text is None  # critical semantic guarantee

    pj = json.loads(proposal_json_raw)
    assert pj["outcome"] == "no_changes"
    assert "no persistent worktree changes" in (pj.get("note") or "").lower()


def test_safe_autofix_no_change_is_not_error(tmp_path: Path, db: SQLiteConn, monkeypatch):
    monkeypatch.setattr(settings, "self_improve_safe_autofix_only", True, raising=False)
    monkeypatch.setattr(
        settings, "self_improve_enable_llm_autonomous_changes", False, raising=False
    )
    monkeypatch.setattr(settings, "github_default_branch", "feature/flywheel", raising=False)

    # No health improvement: baseline=0 gates and after=0 gates, same comparison score => no_change.
    sb = ScoreboardStub(
        repo=tmp_path,
        results=[
            FakeScoreboardResult(gates_failing=0, score_value=10.0),  # baseline
            FakeScoreboardResult(gates_failing=0, score_value=10.0),  # before
            FakeScoreboardResult(gates_failing=0, score_value=10.0),  # after (fix=True)
            FakeScoreboardResult(gates_failing=0, score_value=10.0),  # final
        ],
    )

    ctl = _mk_controller(tmp_path, db, sb)

    budget = SimpleNamespace(
        max_iterations=1,
        max_seconds=999,
        gap_limit=5,
        open_pr_on_improvement=False,
        stop_on_first_improvement=False,
    )

    ctl.run(goal="test", budget=budget, status_callback=None)

    proposal_json_raw, error_text = _latest_attempt_row(db)
    assert error_text is None  # critical semantic guarantee

    pj = json.loads(proposal_json_raw)
    assert pj["outcome"] == "no_change"
    assert "no measurable improvement" in (pj.get("note") or "").lower()
