from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from base.agents.orchestrator import Orchestrator
from base.database.sqlite import SQLiteConn
from base.self_improve.self_improve_db import (
    insert_improvement_attempt,
    insert_score_run,
)


class _DummySelfImprove:
    def __init__(self, result: dict[str, Any]):
        self.result = result
        self.calls: list[bool] = []

    def run_manual(self, *, cfg, include_dream: bool = False):
        # Record how it was called; return deterministic result.
        self.calls.append(bool(include_dream))
        return self.result


def _mk_orch(db: SQLiteConn, *, run_result: dict[str, Any] | None = None) -> Orchestrator:
    # IMPORTANT: bypass __init__ (no embeddings/qdrant/llm)
    orch = Orchestrator.__new__(Orchestrator)
    orch.db = db

    # Only used by dream run:
    if run_result is None:
        run_result = {"artifacts": {"run_dir": "reports/self_improve/run-TEST"}, "pr_url": None}
    orch.self_improve = _DummySelfImprove(run_result)

    # The command handler uses json.loads; ensure module-level json exists in orchestrator.py
    # (already imported there). Nothing else required.
    return orch


def _insert_minimal_run(db: SQLiteConn) -> int:
    return insert_score_run(
        db.conn,
        run_type="iteration_after",
        mode="all",
        fix_enabled=True,
        git_branch="feature/flywheel",
        git_sha="abcdef1234567890",
        score=123.456,
        passed=True,
        metrics={"gates_failing": 0, "score": 123.456},
    )


def _insert_attempt(
    db: SQLiteConn,
    *,
    iteration: int = 1,
    baseline_run_id: int,
    before_run_id: int,
    after_run_id: int | None,
    branch: str,
    proposal_title: str,
    proposal_json: dict[str, Any] | None,
    pr_url: str | None,
    improved: bool,
    error_text: str | None,
) -> int:
    return insert_improvement_attempt(
        db.conn,
        iteration=iteration,
        baseline_run_id=baseline_run_id,
        before_run_id=before_run_id,
        after_run_id=after_run_id,
        branch=branch,
        proposal_title=proposal_title,
        proposal_json=proposal_json,
        pr_url=pr_url,
        improved=improved,
        error_text=error_text,
    )


def _insert_gap(db: SQLiteConn, *, fingerprint: str, priority: int = 10, status: str = "queued"):
    db.conn.execute(
        """
        INSERT INTO capability_gaps(
          source, fingerprint, requested_capability, observed_failure,
          classification, repro_steps, priority, status, metadata_json
        )
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "scoreboard",
            fingerprint,
            "pass_some_gate",
            "unit test injected gap",
            "gate",
            None,
            int(priority),
            status,
            json.dumps({}, ensure_ascii=False),
        ),
    )
    db.conn.commit()


def test_dream_usage_when_missing_subcommand(tmp_path: Path):
    db = SQLiteConn(str(tmp_path / "dream_usage.db"))
    orch = _mk_orch(db)

    out = orch._handle_dream_command("dream")
    assert out is not None
    assert out.startswith("Usage:")


def test_dream_status_shows_note_for_legacy_no_worktree_change_error(tmp_path: Path):
    db = SQLiteConn(str(tmp_path / "dream_status.db"))
    orch = _mk_orch(db)

    run_id = _insert_minimal_run(db)

    # Create an attempt that mimics historical behavior: stored in error_text
    _insert_attempt(
        db,
        baseline_run_id=run_id,
        before_run_id=run_id,
        after_run_id=run_id,
        branch="repo-janitor-autofix-legacy-it1",
        proposal_title="Repo Janitor: safe autofix",
        proposal_json=None,
        pr_url=None,
        improved=False,
        error_text="safe autofix improved scoring but produced no persistent worktree changes",
    )

    out = orch._handle_dream_command("dream status")
    assert out is not None
    assert "Self-improve status:" in out
    assert "Last score run:" in out
    assert "Last attempt:" in out

    # Legacy message must show as Note, not Error
    assert "Note:" in out
    assert "produced no persistent worktree changes" in out.lower()
    assert "Error: safe autofix improved scoring" not in out


def test_dream_last_reads_outcome_and_note_from_proposal_json(tmp_path: Path):
    db = SQLiteConn(str(tmp_path / "dream_last.db"))
    orch = _mk_orch(db)

    run_id = _insert_minimal_run(db)

    _insert_attempt(
        db,
        baseline_run_id=run_id,
        before_run_id=run_id,
        after_run_id=run_id,
        branch="repo-janitor-autofix-it1",
        proposal_title="Repo Janitor: safe autofix",
        proposal_json={
            "outcome": "no_change",
            "note": "Safe autofix made no measurable improvement",
        },
        pr_url=None,
        improved=False,
        error_text=None,
    )

    out = orch._handle_dream_command("dream last")
    assert out is not None
    assert out.startswith("Last self-improve attempt:")
    assert "Outcome: no_change" in out
    assert "Note: Safe autofix made no measurable improvement" in out


def test_dream_attempts_formats_outcome_and_error_suffix(tmp_path: Path):
    db = SQLiteConn(str(tmp_path / "dream_attempts.db"))
    orch = _mk_orch(db)

    run_id = _insert_minimal_run(db)

    # 1) success-ish no_change
    _insert_attempt(
        db,
        baseline_run_id=run_id,
        before_run_id=run_id,
        after_run_id=run_id,
        branch="b1",
        proposal_title="T1",
        proposal_json={"outcome": "no_change"},
        pr_url=None,
        improved=False,
        error_text=None,
    )

    # 2) legacy no-worktree-change stored as error_text
    _insert_attempt(
        db,
        baseline_run_id=run_id,
        before_run_id=run_id,
        after_run_id=run_id,
        branch="b2",
        proposal_title="T2",
        proposal_json=None,
        pr_url=None,
        improved=False,
        error_text="Safe autofix improved scoring, but produced no persistent worktree changes",
    )

    # 3) real error
    _insert_attempt(
        db,
        baseline_run_id=run_id,
        before_run_id=run_id,
        after_run_id=None,
        branch="b3",
        proposal_title="T3",
        proposal_json=None,
        pr_url=None,
        improved=False,
        error_text="boom",
    )

    out = orch._handle_dream_command("dream attempts 5")
    assert out is not None
    assert out.startswith("Last")
    assert "(no_change)" in out
    assert "(no_changes)" in out  # legacy mapping
    assert "(error)" in out


def test_dream_gaps_and_gap_alias(tmp_path: Path):
    db = SQLiteConn(str(tmp_path / "dream_gaps.db"))
    orch = _mk_orch(db)

    # No gaps → friendly empty response
    out0 = orch._handle_dream_command("dream gaps")
    assert out0 == "No open gaps (queued/in_progress/new)."

    # Insert a gap → list appears
    _insert_gap(db, fingerprint="fp-1", priority=25, status="queued")

    out1 = orch._handle_dream_command("dream gaps 15")
    assert out1 is not None
    assert out1.startswith("Open gaps")
    assert "prio=25" in out1

    # Alias should work
    out2 = orch._handle_dream_command("dream gap 15")
    assert out2 is not None
    assert out2.startswith("Open gaps")


def test_dream_run_returns_report_dir(tmp_path: Path):
    db = SQLiteConn(str(tmp_path / "dream_run.db"))
    orch = _mk_orch(
        db,
        run_result={
            "artifacts": {"run_dir": "reports/self_improve/run-XYZ"},
            "pr_url": None,
        },
    )

    out = orch._handle_dream_command("dream run")
    assert out is not None
    assert "Self-improve run complete." in out
    assert "Report dir: reports/self_improve/run-XYZ" in out
