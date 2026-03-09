from __future__ import annotations

import json
import shutil
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

from loguru import logger

from base.self_improve.models import Proposal
from base.self_improve.scoreboard import ScoreboardRunner
from base.self_improve.self_improve_db import (
    fetch_open_gaps,
    insert_improvement_attempt,
    insert_score_run,
    make_gap_fingerprint,
    mark_gap_status,
    reconcile_gap_states_for_source,
    upsert_gap,
)
from config.config import settings


@dataclass(frozen=True)
class IterationBudget:
    max_iterations: int = 3
    max_seconds: int = 20 * 60
    stop_on_first_improvement: bool = True
    open_pr_on_improvement: bool = True
    gap_limit: int = 5


@dataclass
class LeashPolicy:
    """
    Controls which repo paths RepoJanitor is allowed to touch.

    allowlist: if non-empty, ONLY these patterns are permitted
    blocklist: if a path matches any blocklist pattern, it is denied
    """

    allowlist: list[str] = field(default_factory=list)
    blocklist: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Normalize patterns to forward slashes for consistent matching across OSes
        self.allowlist = [p.replace("\\", "/") for p in (self.allowlist or [])]
        self.blocklist = [p.replace("\\", "/") for p in (self.blocklist or [])]

    def is_allowed(self, relpath: str) -> bool:
        rp = (relpath or "").replace("\\", "/")

        # Blocklist wins
        if self.blocklist and any(fnmatch(rp, pat) for pat in self.blocklist):
            return False

        # If allowlist is provided, require match
        if self.allowlist:
            return any(fnmatch(rp, pat) for pat in self.allowlist)

        # No allowlist means allow by default (except blocklist)
        return True


class RepoJanitorIterationController:
    """
    Core loop:
      - baseline scoreboard
      - log gaps
      - iterate: propose -> leash-check -> apply -> leash-check(worktree) -> scoreboard -> improve? -> PR
      - persist repo_score_runs + repo_improvement_attempts each iteration
      - rollback via git restore + checkout AND safe removal of untracked files
    """

    def __init__(
        self,
        *,
        repo_root: str | Path,
        db_conn,
        code_indexer,
        proposal_engine,
        pr_manager,
        policy: LeashPolicy,
    ):
        self.repo = Path(repo_root).resolve()
        # self.db = db_conn
        # self.conn = db_conn.conn
        self.conn = getattr(db_conn, "conn", db_conn)
        self.db = self.conn
        self.code_indexer = code_indexer
        self.proposal_engine = proposal_engine
        self.pr_manager = pr_manager
        self.policy = policy
        self.scoreboard = ScoreboardRunner(self.repo)

    # ---------------- git helpers ----------------

    def _git(self, args: list[str], check: bool = True) -> subprocess.CompletedProcess:
        p = subprocess.run(
            ["git", *args],
            cwd=str(self.repo),
            text=True,
            capture_output=True,
        )
        if check and p.returncode != 0:
            raise RuntimeError(
                f"git {' '.join(args)} failed\nstdout:\n{p.stdout}\nstderr:\n{p.stderr}"
            )
        return p

    def _current_branch(self) -> str:
        return (
            self._git(["rev-parse", "--abbrev-ref", "HEAD"], check=False).stdout.strip() or "HEAD"
        )

    def _current_sha(self) -> str:
        return self._git(["rev-parse", "HEAD"], check=False).stdout.strip()

    def _worktree_dirty(self) -> bool:
        out = self._git(["status", "--porcelain"], check=False).stdout or ""
        return bool(out.strip())

    def _changed_paths(self) -> list[str]:
        out = self._git(["status", "--porcelain"], check=False).stdout
        paths: list[str] = []
        for ln in (out or "").splitlines():
            if not ln.strip():
                continue
            # format: XY <path> or ?? <path>
            paths.append(ln[3:].strip())
        return paths

    def _untracked_paths(self) -> list[str]:
        out = self._git(["status", "--porcelain"], check=False).stdout or ""
        items: list[str] = []
        for ln in out.splitlines():
            if ln.startswith("?? "):
                items.append(ln[3:].strip())
        return items

    def _safe_remove_path(self, relpath: str) -> None:
        """
        Remove a file/dir if it exists. Used only for untracked files produced by an iteration
        and only when within leash allowlist.
        """
        p = (self.repo / relpath).resolve()
        try:
            # Safety: ensure it's inside repo root
            p.relative_to(self.repo)
        except Exception:
            return

        if not p.exists():
            return
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        else:
            try:
                p.unlink()
            except Exception:
                pass

    def _rollback_to_base(self, base_branch: str) -> None:
        """
        Restore working tree + staged state, remove any untracked files created by the attempt (allowlist-only),
        then checkout base branch.
        """
        # restore tracked files
        self._git(["restore", "--staged", "--worktree", "."], check=False)

        # remove untracked produced during attempt (SAFE: allowlist-only)
        for rel in self._untracked_paths():
            if self.policy.is_allowed(rel):
                self._safe_remove_path(rel)

        # switch back
        self._git(["checkout", base_branch], check=False)

        # final restore pass (covers cases where checkout left modified tracked files)
        self._git(["restore", "--staged", "--worktree", "."], check=False)

        # re-remove untracked (some tools may recreate files on checkout hooks)
        for rel in self._untracked_paths():
            if self.policy.is_allowed(rel):
                self._safe_remove_path(rel)

    # ---------------- index md ----------------

    def _index_to_markdown(self, index: Any) -> str:
        if isinstance(index, dict) and "files" in index and isinstance(index["files"], dict):
            lines = ["# Repository Index", ""]
            for path in sorted(index["files"].keys()):
                lines.append(f"- `{path}`")
            return "\n".join(lines)

        to_md = getattr(self.code_indexer, "to_markdown", None)
        if callable(to_md):
            try:
                return to_md(index)
            except Exception:
                pass

        exts = {".py", ".md", ".toml", ".yml", ".yaml", ".json", ".txt", ".ini"}
        ignore = {".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache"}
        lines = ["# Repository Index", ""]
        for p in sorted(self.repo.rglob("*")):
            if any(part in ignore for part in p.parts):
                continue
            if p.is_file() and p.suffix.lower() in exts:
                lines.append(f"- `{p.relative_to(self.repo).as_posix()}`")
        return "\n".join(lines)

    # ---------------- gaps ----------------

    def _log_gaps_from_scoreboard(self, run, source: str = "scoreboard") -> set[str]:
        active_fingerprints: set[str] = set()

        for name, tr in run.tool_results.items():
            if tr.exit_code == 0:
                continue

            tail = (tr.stderr_tail or tr.stdout_tail or "")[:4000]
            fingerprint = make_gap_fingerprint(source, name, str(tr.exit_code), tail)
            active_fingerprints.add(fingerprint)

            upsert_gap(
                self.conn,
                source=source,
                fingerprint=fingerprint,
                requested_capability=f"pass_{name}_gate",
                observed_failure=tail,
                classification="quality_gate",
                repro_steps=f"Scoreboard tool '{name}' failed (exit={tr.exit_code}).",
                priority=50 if name in ("pytest", "compile") else 25,
                metadata={"tool": name, "exit_code": tr.exit_code},
            )

        return active_fingerprints

    def _goal_from_gaps(self, limit: int) -> tuple[str, list[dict[str, Any]]]:
        gaps = fetch_open_gaps(self.conn, limit=limit)
        if not gaps:
            return "Reduce failing gates and improve repository hygiene.", []

        lines = ["Fix the highest priority open capability gaps:"]
        selected_gaps: list[dict[str, Any]] = []

        for gap in gaps:
            selected_gaps.append(
                {
                    "id": int(gap["id"]),
                    "source": str(gap.get("source") or ""),
                    "fingerprint": str(gap.get("fingerprint") or ""),
                }
            )
            lines.append(
                f"- [{gap['classification']}] {gap['requested_capability']} (priority={gap['priority']})"
            )
            if gap.get("observed_failure"):
                tail = (gap["observed_failure"] or "").splitlines()[-10:]
                lines.append("  - failure_tail: " + " | ".join(tail)[:400])

        return "\n".join(lines), selected_gaps

    # ---------------- leash ----------------

    def _enforce_leash_on_proposal(self, proposal) -> tuple[bool, str]:
        bad: list[str] = []
        for ch in getattr(proposal, "changes", []) or []:
            p = getattr(ch, "path", "") or ""
            if not p or not self.policy.is_allowed(p):
                bad.append(p or "(empty)")
        if bad:
            return False, f"Leash policy blocked proposal paths: {', '.join(bad)}"
        return True, "ok"

    def _enforce_leash_on_worktree(self) -> tuple[bool, str]:
        """
        After applying changes, re-check actual changed/untracked paths.
        This prevents writing files outside leash even if the proposal lied or apply logic is buggy.
        """
        bad: list[str] = []
        for p in self._changed_paths():
            if p and not self.policy.is_allowed(p):
                bad.append(p)
        if bad:
            return False, f"Leash policy blocked worktree changes: {', '.join(bad)}"
        return True, "ok"

    def _artifact_relpath(self, path: Path) -> str:
        resolved = path.resolve()
        try:
            return resolved.relative_to(self.repo).as_posix()
        except ValueError:
            return str(resolved)

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _failing_tools(self, run) -> list[str]:
        return sorted(
            name for name, tr in run.tool_results.items() if int(getattr(tr, "exit_code", 1)) != 0
        )

    def _score_delta_summary(self, before, after) -> dict[str, Any]:
        before_failing = set(self._failing_tools(before))
        after_failing = set(self._failing_tools(after))

        return {
            "before_score": float(before.score()),
            "after_score": float(after.score()),
            "score_delta": float(after.score() - before.score()),
            "before_gates": int(before.gates_failing),
            "after_gates": int(after.gates_failing),
            "gates_delta": int(after.gates_failing) - int(before.gates_failing),
            "newly_failing_tools": sorted(after_failing - before_failing),
            "resolved_tools": sorted(before_failing - after_failing),
            "still_failing_tools": sorted(before_failing & after_failing),
        }

    def _write_attempt_summary_artifact(
        self,
        *,
        run_artifact_dir: Path,
        iteration: int,
        branch: str,
        stage: str,
        improved: bool | None,
        error: str | None,
        attempt_artifacts: dict[str, Any],
        extra: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        payload = {
            "schema_version": 1,
            "iteration": int(iteration),
            "branch": branch,
            "stage": stage,
            "improved": improved,
            "error": error,
            "artifacts": attempt_artifacts,
            "extra": extra or {},
        }

        path = run_artifact_dir / f"iteration_{iteration:02d}_attempt_summary.json"
        self._write_json(path, payload)

        return {
            "path": str(path.resolve()),
            "relative_path": self._artifact_relpath(path),
        }

    def _emit_status(
        self,
        callback: Callable[[dict[str, Any]], None] | None,
        *,
        phase: str,
        state: str,
        message: str,
        pct: int | None = None,
        iteration: int | None = None,
        branch: str | None = None,
        duration_ms: float | None = None,
        outcome: str | None = None,
        error: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        if callback is None:
            return

        payload: dict[str, Any] = {
            "phase": phase,
            "state": state,
            "message": message,
            "pct": pct,
            "iteration": iteration,
            "branch": branch,
            "duration_ms": duration_ms,
            "outcome": outcome,
            "error": error,
        }
        if extra:
            payload["extra"] = extra

        try:
            callback(payload)
        except Exception as e:
            logger.debug(f"[self-improve] status callback failed: {e}")

    def _status_start(
        self,
        callback: Callable[[dict[str, Any]], None] | None,
        timers: dict[str, float],
        *,
        phase: str,
        message: str,
        pct: int | None = None,
        iteration: int | None = None,
        branch: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        key = f"{phase}:{iteration if iteration is not None else 0}"
        timers[key] = time.perf_counter()
        self._emit_status(
            callback,
            phase=phase,
            state="start",
            message=message,
            pct=pct,
            iteration=iteration,
            branch=branch,
            extra=extra,
        )

    def _status_finish(
        self,
        callback: Callable[[dict[str, Any]], None] | None,
        timers: dict[str, float],
        *,
        phase: str,
        message: str,
        pct: int | None = None,
        iteration: int | None = None,
        branch: str | None = None,
        outcome: str | None = None,
        error: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        key = f"{phase}:{iteration if iteration is not None else 0}"
        started_at = timers.pop(key, None)

        duration_ms: float | None = None
        if started_at is not None:
            duration_ms = round((time.perf_counter() - started_at) * 1000.0, 2)

        self._emit_status(
            callback,
            phase=phase,
            state="error" if error else "complete",
            message=message,
            pct=pct,
            iteration=iteration,
            branch=branch,
            duration_ms=duration_ms,
            outcome=outcome,
            error=error,
            extra=extra,
        )

    # ---------------- run ----------------

    def run(
        self,
        *,
        goal: str,
        budget: IterationBudget,
        status_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        status_timers: dict[str, float] = {}
        run_token = f"run-{int(time.time() * 1000)}"
        run_artifact_dir = self.repo / "reports" / "self_improve" / run_token
        artifacts: dict[str, Any] = {
            "run_dir": self._artifact_relpath(run_artifact_dir),
            "baseline": None,
            "attempts": [],
            "final": None,
        }

        base_branch = self._current_branch()
        logger.info(f"[self-improve] base_branch={base_branch}")

        user_branch = base_branch
        base_for_branches = (settings.github_default_branch or base_branch or "main").strip()

        if user_branch != base_for_branches:
            logger.info(
                f"[self-improve] switching from {user_branch} -> {base_for_branches} for run base"
            )
            self._git(["checkout", base_for_branches], check=False)
            base_branch = base_for_branches

        if self._worktree_dirty():
            msg = "Working tree is not clean; commit/stash changes before self-improve."
            logger.error(f"[self-improve] {msg}")
            self._emit_status(
                status_callback,
                phase="summarize",
                state="error",
                message="Self-improve run aborted",
                pct=100,
                branch=base_branch,
                error=msg,
            )
            return {
                "goal": goal,
                "baseline": {"score": 0.0, "gates": 0},
                "best": {"score": 0.0, "gates": 0},
                "attempts": [{"iteration": 0, "error": msg}],
                "improved": False,
                "pr_url": None,
                "branch": base_branch,
                "artifacts": artifacts,
            }

        self._status_start(
            status_callback,
            status_timers,
            phase="diagnose",
            message="Running baseline scoreboard",
            pct=5,
            branch=base_branch,
        )

        baseline_artifact = run_artifact_dir / "baseline_scoreboard.json"
        baseline = self.scoreboard.run(
            mode="all",
            fix=False,
            artifact_path=baseline_artifact,
            context={
                "phase": "baseline",
                "branch": base_branch,
                "goal_hint_present": bool(goal.strip()),
            },
        )
        artifacts["baseline"] = {
            "path": baseline.artifact_path,
            "relative_path": baseline.artifact_relpath,
        }

        baseline_id = insert_score_run(
            self.conn,
            run_type="baseline",
            mode="all",
            fix_enabled=False,
            git_branch=base_branch,
            git_sha=self._current_sha(),
            score=float(baseline.score()),
            passed=bool(baseline.passed()),
            metrics=baseline.to_dict(),
        )

        baseline_fingerprints = self._log_gaps_from_scoreboard(baseline, source="scoreboard")
        reconcile_gap_states_for_source(
            self.conn,
            source="scoreboard",
            active_fingerprints=baseline_fingerprints,
            active_status="queued",
        )

        self._status_finish(
            status_callback,
            status_timers,
            phase="diagnose",
            message="Baseline scoreboard recorded",
            pct=15,
            branch=base_branch,
            outcome=f"gates={int(baseline.gates_failing)} score={float(baseline.score()):.2f}",
        )

        self._status_start(
            status_callback,
            status_timers,
            phase="hypothesize",
            message="Building goal from open gaps",
            pct=20,
            branch=base_branch,
        )

        computed_goal, selected_gaps = self._goal_from_gaps(limit=budget.gap_limit)
        if goal.strip():
            computed_goal = goal.strip() + "\n\n" + computed_goal

        for gap in selected_gaps:
            mark_gap_status(self.conn, int(gap["id"]), "in_progress")

        self._status_finish(
            status_callback,
            status_timers,
            phase="hypothesize",
            message="Improvement goal prepared",
            pct=30,
            branch=base_branch,
            outcome=f"{len(selected_gaps)} tracked gaps",
        )

        best = baseline
        best_id = baseline_id
        attempts: list[dict[str, Any]] = []
        improved_any = False
        pr_url: str | None = None
        best_branch: str | None = None
        safe_autofix_only = bool(getattr(settings, "self_improve_safe_autofix_only", True))
        allow_llm_autonomous = bool(
            getattr(settings, "self_improve_enable_llm_autonomous_changes", False)
        )

        for i in range(1, budget.max_iterations + 1):
            attempt_artifacts: dict[str, Any] = {
                "iteration": i,
                "before": None,
                "after": None,
                "diff_summary": None,
                "summary": None,
            }

            if (time.perf_counter() - started) > budget.max_seconds:
                attempts.append({"iteration": i, "error": "budget timeout"})
                attempt_artifacts["summary"] = self._write_attempt_summary_artifact(
                    run_artifact_dir=run_artifact_dir,
                    iteration=i,
                    branch=base_branch,
                    stage="timeout",
                    improved=False,
                    error="budget timeout",
                    attempt_artifacts=attempt_artifacts,
                    extra={},
                )
                artifacts["attempts"].append(attempt_artifacts)
                insert_improvement_attempt(
                    self.conn,
                    iteration=i,
                    baseline_run_id=baseline_id,
                    before_run_id=best_id,
                    after_run_id=None,
                    branch=base_branch,
                    proposal_title=None,
                    proposal_json={"timeout": True, "artifacts": attempt_artifacts},
                    pr_url=None,
                    improved=False,
                    error_text="budget timeout",
                )
                self._emit_status(
                    status_callback,
                    phase="summarize",
                    state="error",
                    message="Self-improve run timed out",
                    pct=100,
                    iteration=i,
                    branch=base_branch,
                    error="budget timeout",
                )
                break

            if i == 1:
                branch_name = f"repo-janitor-autofix-{int(time.time())}-it{i}"
                branch = self.pr_manager.prepare_branch(branch_name, base=base_for_branches)

                self._status_start(
                    status_callback,
                    status_timers,
                    phase="patch",
                    message="Running safe autofix lane",
                    pct=45,
                    iteration=i,
                    branch=branch,
                )

                before_artifact = run_artifact_dir / f"iteration_{i:02d}_before_scoreboard.json"
                before = self.scoreboard.run(
                    mode="all",
                    fix=False,
                    artifact_path=before_artifact,
                    context={
                        "phase": "iteration_before_autofix",
                        "iteration": i,
                        "branch": branch,
                        "attempt_type": "safe_autofix",
                    },
                )
                attempt_artifacts["before"] = {
                    "path": before.artifact_path,
                    "relative_path": before.artifact_relpath,
                }

                before_id = insert_score_run(
                    self.conn,
                    run_type="iteration_before",
                    mode="all",
                    fix_enabled=False,
                    git_branch=branch,
                    git_sha=self._current_sha(),
                    score=float(before.score()),
                    passed=bool(before.passed()),
                    metrics=before.to_dict(),
                )

                after_artifact = run_artifact_dir / f"iteration_{i:02d}_after_scoreboard.json"
                after = self.scoreboard.run(
                    mode="all",
                    fix=True,
                    artifact_path=after_artifact,
                    context={
                        "phase": "iteration_after_autofix",
                        "iteration": i,
                        "branch": branch,
                        "attempt_type": "safe_autofix",
                    },
                )
                attempt_artifacts["after"] = {
                    "path": after.artifact_path,
                    "relative_path": after.artifact_relpath,
                }

                after_id = insert_score_run(
                    self.conn,
                    run_type="iteration_after",
                    mode="all",
                    fix_enabled=True,
                    git_branch=branch,
                    git_sha=self._current_sha(),
                    score=float(after.score()),
                    passed=bool(after.passed()),
                    metrics=after.to_dict(),
                )
                self._log_gaps_from_scoreboard(after, source="scoreboard")

                patch_outcome = (
                    "changes_detected" if self._worktree_dirty() else "no_worktree_changes"
                )
                self._status_finish(
                    status_callback,
                    status_timers,
                    phase="patch",
                    message="Safe autofix lane finished",
                    pct=60,
                    iteration=i,
                    branch=branch,
                    outcome=patch_outcome,
                )

                self._status_start(
                    status_callback,
                    status_timers,
                    phase="verify",
                    message="Comparing safe autofix results",
                    pct=65,
                    iteration=i,
                    branch=branch,
                )

                diff_summary = self._score_delta_summary(before, after)
                diff_summary.update(
                    {
                        "iteration": i,
                        "branch": branch,
                        "proposal_title": "Repo Janitor: safe autofix",
                        "attempt_type": "safe_autofix",
                        "before_artifact": before.artifact_relpath,
                        "after_artifact": after.artifact_relpath,
                    }
                )

                diff_path = run_artifact_dir / f"iteration_{i:02d}_diff_summary.json"
                self._write_json(diff_path, diff_summary)
                attempt_artifacts["diff_summary"] = {
                    "path": str(diff_path.resolve()),
                    "relative_path": self._artifact_relpath(diff_path),
                }

                best_gates = int(getattr(best, "gates_failing", 0))
                after_gates = int(getattr(after, "gates_failing", 0))
                best_score = float(best.score())
                after_score = float(after.score())
                eps = 1e-6

                if after_gates < best_gates:
                    is_improved = True
                elif after_gates > best_gates:
                    is_improved = False
                else:
                    is_improved = after_score > (best_score + eps)

                is_improved = bool(is_improved and self._worktree_dirty())

                self._status_finish(
                    status_callback,
                    status_timers,
                    phase="verify",
                    message=(
                        "Safe autofix improved the scoreboard"
                        if is_improved
                        else "Safe autofix made no measurable improvement"
                    ),
                    pct=80,
                    iteration=i,
                    branch=branch,
                    outcome="improved" if is_improved else "no_change",
                )

                attempt_artifacts["summary"] = self._write_attempt_summary_artifact(
                    run_artifact_dir=run_artifact_dir,
                    iteration=i,
                    branch=branch,
                    stage="safe_autofix_scored",
                    improved=bool(is_improved),
                    error=None if is_improved else "safe autofix made no measurable improvement",
                    attempt_artifacts=attempt_artifacts,
                    extra={"diff_summary": diff_summary},
                )

                attempt_row = {
                    "iteration": i,
                    "branch": branch,
                    "before_score": float(before.score()),
                    "after_score": float(after.score()),
                    "before_gates": int(before.gates_failing),
                    "after_gates": int(after.gates_failing),
                    "improved": bool(is_improved),
                    "mode": "safe_autofix",
                    "artifacts": attempt_artifacts,
                    "diff_summary": diff_summary,
                }
                attempts.append(attempt_row)
                artifacts["attempts"].append(attempt_artifacts)

                if is_improved:
                    improved_any = True
                    best = after
                    best_id = after_id
                    best_branch = branch

                    safe_autofix_proposal = Proposal(
                        title="Repo Janitor: safe autofix",
                        description="Applied Black formatting and Ruff safe fixes.",
                        changes=[],
                    )

                    pr_error: str | None = None
                    pr_url = None

                    if budget.open_pr_on_improvement:
                        self._status_start(
                            status_callback,
                            status_timers,
                            phase="propose_pr",
                            message="Opening improvement PR",
                            pct=85,
                            iteration=i,
                            branch=branch,
                        )
                        try:
                            self.pr_manager.commit_and_push(branch, safe_autofix_proposal.title)
                            pr_url = self.pr_manager.open_pr(
                                branch=branch, proposal=safe_autofix_proposal
                            )
                            if pr_url:
                                try:
                                    self.pr_manager.run_tests_and_update_pr(branch)
                                except Exception:
                                    pass

                            self._status_finish(
                                status_callback,
                                status_timers,
                                phase="propose_pr",
                                message="Improvement PR ready",
                                pct=95,
                                iteration=i,
                                branch=branch,
                                outcome=pr_url or "branch_pushed",
                            )
                        except Exception as e:
                            pr_error = str(e)
                            self._status_finish(
                                status_callback,
                                status_timers,
                                phase="propose_pr",
                                message="Improvement PR failed",
                                pct=95,
                                iteration=i,
                                branch=branch,
                                error=pr_error,
                            )

                    insert_improvement_attempt(
                        self.conn,
                        iteration=i,
                        baseline_run_id=baseline_id,
                        before_run_id=before_id,
                        after_run_id=after_id,
                        branch=branch,
                        proposal_title=safe_autofix_proposal.title,
                        proposal_json={
                            "mode": "safe_autofix",
                            "description": safe_autofix_proposal.description,
                            "artifacts": attempt_artifacts,
                            "diff_summary": diff_summary,
                        },
                        pr_url=pr_url,
                        improved=True,
                        error_text=pr_error,
                    )

                    if budget.stop_on_first_improvement:
                        break

                insert_improvement_attempt(
                    self.conn,
                    iteration=i,
                    baseline_run_id=baseline_id,
                    before_run_id=before_id,
                    after_run_id=after_id,
                    branch=branch,
                    proposal_title="Repo Janitor: safe autofix",
                    proposal_json={
                        "mode": "safe_autofix",
                        "description": "Applied Black formatting and Ruff safe fixes.",
                        "artifacts": attempt_artifacts,
                        "diff_summary": diff_summary,
                    },
                    pr_url=None,
                    improved=False,
                    error_text="safe autofix made no measurable improvement",
                )
                self._rollback_to_base(base_branch)

                if safe_autofix_only and not allow_llm_autonomous:
                    break

                continue

            if safe_autofix_only and not allow_llm_autonomous:
                msg = "LLM autonomous changes disabled by policy after safe autofix lane"
                attempts.append(
                    {
                        "iteration": i,
                        "error": msg,
                        "mode": "policy_stop",
                    }
                )
                attempt_artifacts["summary"] = self._write_attempt_summary_artifact(
                    run_artifact_dir=run_artifact_dir,
                    iteration=i,
                    branch=base_branch,
                    stage="policy_stop",
                    improved=False,
                    error=msg,
                    attempt_artifacts=attempt_artifacts,
                    extra={"safe_autofix_only": True, "allow_llm_autonomous": False},
                )
                artifacts["attempts"].append(attempt_artifacts)
                insert_improvement_attempt(
                    self.conn,
                    iteration=i,
                    baseline_run_id=baseline_id,
                    before_run_id=best_id,
                    after_run_id=None,
                    branch=base_branch,
                    proposal_title=None,
                    proposal_json={
                        "mode": "policy_stop",
                        "artifacts": attempt_artifacts,
                    },
                    pr_url=None,
                    improved=False,
                    error_text=msg,
                )
                self._emit_status(
                    status_callback,
                    phase="hypothesize",
                    state="error",
                    message="LLM autonomous changes disabled",
                    pct=40,
                    iteration=i,
                    branch=base_branch,
                    error=msg,
                )
                break

            self._status_start(
                status_callback,
                status_timers,
                phase="hypothesize",
                message="Generating proposal from repository gaps",
                pct=35,
                iteration=i,
                branch=base_branch,
            )

            idx = self.code_indexer.scan(incremental=True)  # type: ignore[call-arg]
            index_md = self._index_to_markdown(idx)

            proposal = self.proposal_engine.propose(computed_goal, index_md=index_md)

            ok, msg = self._enforce_leash_on_proposal(proposal)
            if not ok:
                attempts.append({"iteration": i, "error": msg})
                self._status_finish(
                    status_callback,
                    status_timers,
                    phase="hypothesize",
                    message="Proposal rejected by leash policy",
                    pct=40,
                    iteration=i,
                    branch=base_branch,
                    error=msg,
                )
                attempt_artifacts["summary"] = self._write_attempt_summary_artifact(
                    run_artifact_dir=run_artifact_dir,
                    iteration=i,
                    branch=base_branch,
                    stage="proposal_leash_blocked",
                    improved=False,
                    error=msg,
                    attempt_artifacts=attempt_artifacts,
                    extra={"proposal_title": getattr(proposal, "title", None)},
                )
                artifacts["attempts"].append(attempt_artifacts)
                insert_improvement_attempt(
                    self.conn,
                    iteration=i,
                    baseline_run_id=baseline_id,
                    before_run_id=best_id,
                    after_run_id=None,
                    branch=base_branch,
                    proposal_title=getattr(proposal, "title", None),
                    proposal_json={
                        "title": getattr(proposal, "title", ""),
                        "blocked": True,
                        "artifacts": attempt_artifacts,
                    },
                    pr_url=None,
                    improved=False,
                    error_text=msg,
                )
                continue

            if not getattr(proposal, "changes", None):
                msg = "empty proposal (no changes)"
                attempts.append({"iteration": i, "error": msg})
                self._status_finish(
                    status_callback,
                    status_timers,
                    phase="hypothesize",
                    message="Proposal contained no changes",
                    pct=40,
                    iteration=i,
                    branch=base_branch,
                    error=msg,
                )
                attempt_artifacts["summary"] = self._write_attempt_summary_artifact(
                    run_artifact_dir=run_artifact_dir,
                    iteration=i,
                    branch=base_branch,
                    stage="empty_proposal",
                    improved=False,
                    error=msg,
                    attempt_artifacts=attempt_artifacts,
                    extra={"proposal_title": getattr(proposal, "title", None)},
                )
                artifacts["attempts"].append(attempt_artifacts)
                insert_improvement_attempt(
                    self.conn,
                    iteration=i,
                    baseline_run_id=baseline_id,
                    before_run_id=best_id,
                    after_run_id=None,
                    branch=base_branch,
                    proposal_title=getattr(proposal, "title", None),
                    proposal_json={
                        "title": getattr(proposal, "title", ""),
                        "empty": True,
                        "artifacts": attempt_artifacts,
                    },
                    pr_url=None,
                    improved=False,
                    error_text=msg,
                )
                continue

            applicable, refusals = self.proposal_engine.preflight_proposal(proposal)

            if not applicable:
                msg = "proposal has no applicable changes under current safety policy"
                attempts.append({"iteration": i, "error": msg, "refusals": refusals[:5]})
                self._status_finish(
                    status_callback,
                    status_timers,
                    phase="hypothesize",
                    message="Proposal failed preflight checks",
                    pct=40,
                    iteration=i,
                    branch=base_branch,
                    error=msg,
                    extra={"refusals": refusals[:5]},
                )
                attempt_artifacts["summary"] = self._write_attempt_summary_artifact(
                    run_artifact_dir=run_artifact_dir,
                    iteration=i,
                    branch=base_branch,
                    stage="preflight_refused",
                    improved=False,
                    error=msg,
                    attempt_artifacts=attempt_artifacts,
                    extra={
                        "proposal_title": getattr(proposal, "title", None),
                        "refusals": refusals,
                    },
                )
                artifacts["attempts"].append(attempt_artifacts)
                insert_improvement_attempt(
                    self.conn,
                    iteration=i,
                    baseline_run_id=baseline_id,
                    before_run_id=best_id,
                    after_run_id=None,
                    branch=base_branch,
                    proposal_title=getattr(proposal, "title", None),
                    proposal_json={
                        "title": getattr(proposal, "title", ""),
                        "description": getattr(proposal, "description", ""),
                        "preflight_refusals": refusals,
                        "artifacts": attempt_artifacts,
                    },
                    pr_url=None,
                    improved=False,
                    error_text=msg,
                )
                continue

            proposal.changes = applicable

            self._status_finish(
                status_callback,
                status_timers,
                phase="hypothesize",
                message="Proposal accepted for application",
                pct=40,
                iteration=i,
                branch=base_branch,
                outcome=getattr(proposal, "title", "proposal"),
            )

            branch_name = f"repo-janitor-{int(time.time())}-it{i}"
            branch = self.pr_manager.prepare_branch(branch_name, base=base_for_branches)

            self._status_start(
                status_callback,
                status_timers,
                phase="patch",
                message="Applying proposal changes",
                pct=45,
                iteration=i,
                branch=branch,
            )

            before_artifact = run_artifact_dir / f"iteration_{i:02d}_before_scoreboard.json"
            before = self.scoreboard.run(
                mode="all",
                fix=False,
                artifact_path=before_artifact,
                context={
                    "phase": "iteration_before",
                    "iteration": i,
                    "branch": branch,
                    "proposal_title": getattr(proposal, "title", None),
                },
            )
            attempt_artifacts["before"] = {
                "path": before.artifact_path,
                "relative_path": before.artifact_relpath,
            }

            before_id = insert_score_run(
                self.conn,
                run_type="iteration_before",
                mode="all",
                fix_enabled=False,
                git_branch=branch,
                git_sha=self._current_sha(),
                score=float(before.score()),
                passed=bool(before.passed()),
                metrics=before.to_dict(),
            )

            applied = self.proposal_engine.apply_proposal(proposal)
            applied_ok = any(ok for (_c, ok, _m) in applied)

            if not applied_ok:
                details = "; ".join(
                    f"{getattr(c, 'path', '?')}: {msg}" for (c, ok, msg) in applied if not ok
                )[:800]

                self._rollback_to_base(base_branch)
                msg = f"proposal applied no changes ({details})"
                attempts.append({"iteration": i, "error": msg})
                self._status_finish(
                    status_callback,
                    status_timers,
                    phase="patch",
                    message="Proposal application failed",
                    pct=60,
                    iteration=i,
                    branch=branch,
                    error=msg,
                )
                attempt_artifacts["summary"] = self._write_attempt_summary_artifact(
                    run_artifact_dir=run_artifact_dir,
                    iteration=i,
                    branch=branch,
                    stage="apply_no_changes",
                    improved=False,
                    error=msg,
                    attempt_artifacts=attempt_artifacts,
                    extra={
                        "proposal_title": getattr(proposal, "title", None),
                        "details": details,
                    },
                )
                artifacts["attempts"].append(attempt_artifacts)
                insert_improvement_attempt(
                    self.conn,
                    iteration=i,
                    baseline_run_id=baseline_id,
                    before_run_id=before_id,
                    after_run_id=None,
                    branch=branch,
                    proposal_title=getattr(proposal, "title", None),
                    proposal_json={
                        "title": getattr(proposal, "title", ""),
                        "applied": False,
                        "details": details,
                        "artifacts": attempt_artifacts,
                    },
                    pr_url=None,
                    improved=False,
                    error_text=msg,
                )
                continue

            ok2, msg2 = self._enforce_leash_on_worktree()
            if not ok2:
                self._rollback_to_base(base_branch)
                attempts.append({"iteration": i, "error": msg2})
                self._status_finish(
                    status_callback,
                    status_timers,
                    phase="patch",
                    message="Proposal violated worktree leash",
                    pct=60,
                    iteration=i,
                    branch=branch,
                    error=msg2,
                )
                attempt_artifacts["summary"] = self._write_attempt_summary_artifact(
                    run_artifact_dir=run_artifact_dir,
                    iteration=i,
                    branch=branch,
                    stage="worktree_leash_blocked",
                    improved=False,
                    error=msg2,
                    attempt_artifacts=attempt_artifacts,
                    extra={"proposal_title": getattr(proposal, "title", None)},
                )
                artifacts["attempts"].append(attempt_artifacts)
                insert_improvement_attempt(
                    self.conn,
                    iteration=i,
                    baseline_run_id=baseline_id,
                    before_run_id=before_id,
                    after_run_id=None,
                    branch=branch,
                    proposal_title=getattr(proposal, "title", None),
                    proposal_json={
                        "title": getattr(proposal, "title", ""),
                        "blocked_worktree": True,
                        "artifacts": attempt_artifacts,
                    },
                    pr_url=None,
                    improved=False,
                    error_text=msg2,
                )
                continue

            self._status_finish(
                status_callback,
                status_timers,
                phase="patch",
                message="Proposal applied to worktree",
                pct=60,
                iteration=i,
                branch=branch,
                outcome=getattr(proposal, "title", "proposal"),
            )

            self._status_start(
                status_callback,
                status_timers,
                phase="verify",
                message="Verifying proposal results",
                pct=65,
                iteration=i,
                branch=branch,
            )

            after_artifact = run_artifact_dir / f"iteration_{i:02d}_after_scoreboard.json"
            after = self.scoreboard.run(
                mode="all",
                fix=False,
                artifact_path=after_artifact,
                context={
                    "phase": "iteration_after",
                    "iteration": i,
                    "branch": branch,
                    "proposal_title": getattr(proposal, "title", None),
                },
            )
            attempt_artifacts["after"] = {
                "path": after.artifact_path,
                "relative_path": after.artifact_relpath,
            }

            after_id = insert_score_run(
                self.conn,
                run_type="iteration_after",
                mode="all",
                fix_enabled=False,
                git_branch=branch,
                git_sha=self._current_sha(),
                score=float(after.score()),
                passed=bool(after.passed()),
                metrics=after.to_dict(),
            )
            self._log_gaps_from_scoreboard(after, source="scoreboard")

            diff_summary = self._score_delta_summary(before, after)
            diff_summary.update(
                {
                    "iteration": i,
                    "branch": branch,
                    "proposal_title": getattr(proposal, "title", None),
                    "before_artifact": before.artifact_relpath,
                    "after_artifact": after.artifact_relpath,
                }
            )

            diff_path = run_artifact_dir / f"iteration_{i:02d}_diff_summary.json"
            self._write_json(diff_path, diff_summary)
            attempt_artifacts["diff_summary"] = {
                "path": str(diff_path.resolve()),
                "relative_path": self._artifact_relpath(diff_path),
            }

            best_gates = int(getattr(best, "gates_failing", 0))
            after_gates = int(getattr(after, "gates_failing", 0))
            best_score = float(best.score())
            after_score = float(after.score())
            eps = 1e-6

            if after_gates < best_gates:
                is_improved = True
            elif after_gates > best_gates:
                is_improved = False
            else:
                is_improved = after_score > (best_score + eps)

            self._status_finish(
                status_callback,
                status_timers,
                phase="verify",
                message=(
                    "Proposal improved the scoreboard"
                    if is_improved
                    else "Proposal did not improve the scoreboard"
                ),
                pct=80,
                iteration=i,
                branch=branch,
                outcome="improved" if is_improved else "no_change",
            )

            attempt_artifacts["summary"] = self._write_attempt_summary_artifact(
                run_artifact_dir=run_artifact_dir,
                iteration=i,
                branch=branch,
                stage="scored",
                improved=bool(is_improved),
                error=None,
                attempt_artifacts=attempt_artifacts,
                extra={"diff_summary": diff_summary},
            )

            attempt_row = {
                "iteration": i,
                "branch": branch,
                "before_score": float(before.score()),
                "after_score": float(after.score()),
                "before_gates": int(before.gates_failing),
                "after_gates": int(after.gates_failing),
                "improved": bool(is_improved),
                "artifacts": attempt_artifacts,
                "diff_summary": diff_summary,
            }
            attempts.append(attempt_row)
            artifacts["attempts"].append(attempt_artifacts)

            if is_improved:
                improved_any = True
                best = after
                best_id = after_id
                best_branch = branch

                pr_error: str | None = None

                if budget.open_pr_on_improvement:
                    self._status_start(
                        status_callback,
                        status_timers,
                        phase="propose_pr",
                        message="Opening improvement PR",
                        pct=85,
                        iteration=i,
                        branch=branch,
                    )
                    try:
                        self.pr_manager.commit_and_push(
                            branch,
                            getattr(proposal, "title", "Repo Janitor improvement"),
                        )
                        pr_url = self.pr_manager.open_pr(branch=branch, proposal=proposal)
                        try:
                            self.pr_manager.run_tests_and_update_pr(branch)
                        except Exception:
                            pass

                        self._status_finish(
                            status_callback,
                            status_timers,
                            phase="propose_pr",
                            message="Improvement PR ready",
                            pct=95,
                            iteration=i,
                            branch=branch,
                            outcome=pr_url or "branch_pushed",
                        )
                    except Exception as e:
                        pr_error = str(e)
                        self._status_finish(
                            status_callback,
                            status_timers,
                            phase="propose_pr",
                            message="Improvement PR failed",
                            pct=95,
                            iteration=i,
                            branch=branch,
                            error=pr_error,
                        )

                insert_improvement_attempt(
                    self.conn,
                    iteration=i,
                    baseline_run_id=baseline_id,
                    before_run_id=before_id,
                    after_run_id=after_id,
                    branch=branch,
                    proposal_title=getattr(proposal, "title", None),
                    proposal_json={
                        "title": getattr(proposal, "title", ""),
                        "description": getattr(proposal, "description", ""),
                        "artifacts": attempt_artifacts,
                        "diff_summary": diff_summary,
                    },
                    pr_url=pr_url,
                    improved=True,
                    error_text=pr_error,
                )

                if budget.stop_on_first_improvement:
                    break
            else:
                insert_improvement_attempt(
                    self.conn,
                    iteration=i,
                    baseline_run_id=baseline_id,
                    before_run_id=before_id,
                    after_run_id=after_id,
                    branch=branch,
                    proposal_title=getattr(proposal, "title", None),
                    proposal_json={
                        "title": getattr(proposal, "title", ""),
                        "description": getattr(proposal, "description", ""),
                        "artifacts": attempt_artifacts,
                        "diff_summary": diff_summary,
                    },
                    pr_url=None,
                    improved=False,
                    error_text=None,
                )
                self._rollback_to_base(base_branch)

        self._status_start(
            status_callback,
            status_timers,
            phase="summarize",
            message="Finalizing self-improvement run",
            pct=90,
            branch=best_branch or base_branch,
        )

        final_artifact = run_artifact_dir / "final_scoreboard.json"
        final = self.scoreboard.run(
            mode="all",
            fix=False,
            artifact_path=final_artifact,
            context={
                "phase": "final",
                "branch": best_branch or base_branch,
                "improved": bool(improved_any),
            },
        )
        artifacts["final"] = {
            "path": final.artifact_path,
            "relative_path": final.artifact_relpath,
        }

        final_fingerprints = self._log_gaps_from_scoreboard(final, source="scoreboard")
        reconcile_gap_states_for_source(
            self.conn,
            source="scoreboard",
            active_fingerprints=final_fingerprints,
            active_status="queued",
        )

        for gap in selected_gaps:
            if str(gap.get("source") or "") == "scoreboard":
                continue
            mark_gap_status(
                self.conn,
                int(gap["id"]),
                "fixed" if final.passed() else "queued",
            )

        try:
            self._rollback_to_base(base_branch)
        except Exception:
            pass

        try:
            if user_branch != base_branch:
                self._git(["checkout", user_branch], check=False)
        except Exception:
            pass

        self._status_finish(
            status_callback,
            status_timers,
            phase="summarize",
            message="Self-improvement run complete",
            pct=100,
            branch=best_branch or base_branch,
            outcome="improved" if improved_any else "no_change",
        )

        return {
            "goal": computed_goal,
            "baseline": {"score": float(baseline.score()), "gates": int(baseline.gates_failing)},
            "best": {"score": float(best.score()), "gates": int(best.gates_failing)},
            "attempts": attempts,
            "improved": bool(improved_any),
            "pr_url": pr_url,
            "branch": best_branch or base_branch,
            "artifacts": artifacts,
        }
