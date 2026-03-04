from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

from loguru import logger

from base.self_improve.scoreboard import ScoreboardRunner
from base.self_improve.self_improve_db import (
    fetch_open_gaps,
    insert_improvement_attempt,
    insert_score_run,
    make_gap_fingerprint,
    mark_gap_status,
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
        return self._git(["rev-parse", "--abbrev-ref", "HEAD"], check=False).stdout.strip() or "HEAD"

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

    def _log_gaps_from_scoreboard(self, run, source: str = "scoreboard") -> None:
        for name, tr in run.tool_results.items():
            if tr.exit_code == 0:
                continue
            tail = (tr.stderr_tail or tr.stdout_tail or "")[:4000]
            fp = make_gap_fingerprint(source, name, str(tr.exit_code), tail)
            upsert_gap(
                self.conn,
                source=source,
                fingerprint=fp,
                requested_capability=f"pass_{name}_gate",
                observed_failure=tail,
                classification="quality_gate",
                repro_steps=f"Scoreboard tool '{name}' failed (exit={tr.exit_code}).",
                priority=50 if name in ("pytest", "compile") else 25,
                metadata={"tool": name, "exit_code": tr.exit_code},
            )

    def _goal_from_gaps(self, limit: int) -> tuple[str, list[int]]:
        gaps = fetch_open_gaps(self.conn, limit=limit)
        if not gaps:
            return "Reduce failing gates and improve repository hygiene.", []

        lines = ["Fix the highest priority open capability gaps:"]
        ids: list[int] = []
        for g in gaps:
            ids.append(int(g["id"]))
            lines.append(
                f"- [{g['classification']}] {g['requested_capability']} (priority={g['priority']})"
            )
            if g.get("observed_failure"):
                tail = (g["observed_failure"] or "").splitlines()[-10:]
                lines.append("  - failure_tail: " + " | ".join(tail)[:400])
        return "\n".join(lines), ids

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

    # ---------------- run ----------------

    def run(self, *, goal: str, budget: IterationBudget) -> dict[str, Any]:
        started = time.perf_counter()
        base_branch = self._current_branch()
        logger.info(f"[self-improve] base_branch={base_branch}")

        # Normalize run base: work from the PR base branch so scoring + diffs match what will be merged.
        user_branch = base_branch
        base_for_branches = (settings.github_default_branch or base_branch or "main").strip()

        if user_branch != base_for_branches:
            logger.info(f"[self-improve] switching from {user_branch} -> {base_for_branches} for run base")
            self._git(["checkout", base_for_branches], check=False)
            base_branch = base_for_branches
            
        # Hard safety: do not run if user already has a dirty working tree
        if self._worktree_dirty():
            msg = "Working tree is not clean; commit/stash changes before self-improve."
            logger.error(f"[self-improve] {msg}")
            return {
                "goal": goal,
                "baseline": {"score": 0.0, "gates": []},
                "best": {"score": 0.0, "gates": []},
                "attempts": [{"iteration": 0, "error": msg}],
                "improved": False,
                "pr_url": None,
                "branch": base_branch,
            }

        # Baseline
        baseline = self.scoreboard.run(mode="all", fix=False)
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
        self._log_gaps_from_scoreboard(baseline, source="scoreboard")

        computed_goal, gap_ids = self._goal_from_gaps(limit=budget.gap_limit)
        if goal.strip():
            computed_goal = goal.strip() + "\n\n" + computed_goal

        for gid in gap_ids:
            mark_gap_status(self.conn, gid, "in_progress")

        best = baseline
        best_id = baseline_id
        attempts: list[dict[str, Any]] = []
        improved_any = False
        pr_url: str | None = None
        best_branch: str | None = None

        # base_for_branches = settings.github_default_branch or "main"

        for i in range(1, budget.max_iterations + 1):
            if (time.perf_counter() - started) > budget.max_seconds:
                attempts.append({"iteration": i, "error": "budget timeout"})
                insert_improvement_attempt(
                    self.conn,
                    iteration=i,
                    baseline_run_id=baseline_id,
                    before_run_id=best_id,
                    after_run_id=None,
                    branch=base_branch,
                    proposal_title=None,
                    proposal_json={"timeout": True},
                    pr_url=None,
                    improved=False,
                    error_text="budget timeout",
                )
                break

            # Build index_md (via CodeIndexer)
            idx = self.code_indexer.scan(incremental=True)  # type: ignore[call-arg]
            index_md = self._index_to_markdown(idx)

            # Propose patch
            proposal = self.proposal_engine.propose(computed_goal, index_md=index_md)

            ok, msg = self._enforce_leash_on_proposal(proposal)
            if not ok:
                attempts.append({"iteration": i, "error": msg})
                insert_improvement_attempt(
                    self.conn,
                    iteration=i,
                    baseline_run_id=baseline_id,
                    before_run_id=best_id,
                    after_run_id=None,
                    branch=base_branch,
                    proposal_title=getattr(proposal, "title", None),
                    proposal_json={"title": getattr(proposal, "title", ""), "blocked": True},
                    pr_url=None,
                    improved=False,
                    error_text=msg,
                )
                continue

            # Prepare branch (include iteration so you can read history easily)
            branch_name = f"repo-janitor-{int(time.time())}-it{i}"
            branch = self.pr_manager.prepare_branch(branch_name, base=base_for_branches)

            before = self.scoreboard.run(mode="all", fix=False)
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

            # Apply proposal
            applied = self.proposal_engine.apply_proposal(proposal)
            applied_ok = any(ok for (_c, ok, _m) in applied)
            if not applied_ok:
                self._rollback_to_base(base_branch)
                attempts.append({"iteration": i, "error": "proposal applied no changes"})
                insert_improvement_attempt(
                    self.conn,
                    iteration=i,
                    baseline_run_id=baseline_id,
                    before_run_id=before_id,
                    after_run_id=None,
                    branch=branch,
                    proposal_title=getattr(proposal, "title", None),
                    proposal_json={"title": getattr(proposal, "title", ""), "applied": False},
                    pr_url=None,
                    improved=False,
                    error_text="proposal applied no changes",
                )
                continue

            # Enforce leash on actual worktree changes (critical!)
            ok2, msg2 = self._enforce_leash_on_worktree()
            if not ok2:
                self._rollback_to_base(base_branch)
                attempts.append({"iteration": i, "error": msg2})
                insert_improvement_attempt(
                    self.conn,
                    iteration=i,
                    baseline_run_id=baseline_id,
                    before_run_id=before_id,
                    after_run_id=None,
                    branch=branch,
                    proposal_title=getattr(proposal, "title", None),
                    proposal_json={"title": getattr(proposal, "title", ""), "blocked_worktree": True},
                    pr_url=None,
                    improved=False,
                    error_text=msg2,
                )
                continue

            after = self.scoreboard.run(mode="all", fix=False)
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

            is_improved = after.score() > best.score()
            attempt_row = {
                "iteration": i,
                "branch": branch,
                "before_score": float(before.score()),
                "after_score": float(after.score()),
                "before_gates": list(before.gates_failing),
                "after_gates": list(after.gates_failing),
                "improved": bool(is_improved),
            }
            attempts.append(attempt_row)

            if is_improved:
                improved_any = True
                best = after
                best_id = after_id
                best_branch = branch

                # Commit + push + PR
                if budget.open_pr_on_improvement:
                    self.pr_manager.commit_and_push(
                        branch,
                        getattr(proposal, "title", "Repo Janitor improvement"),
                    )
                    pr_url = self.pr_manager.open_pr(branch=branch, proposal=proposal)
                    try:
                        self.pr_manager.run_tests_and_update_pr(branch)
                    except Exception:
                        pass

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
                    },
                    pr_url=pr_url,
                    improved=True,
                    error_text=None,
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
                    },
                    pr_url=None,
                    improved=False,
                    error_text=None,
                )
                self._rollback_to_base(base_branch)

        # Final scoreboard on current state (best branch if improved + stopped; base branch otherwise)
        final = self.scoreboard.run(mode="all", fix=False)

        if improved_any and final.passed():
            for gid in gap_ids:
                mark_gap_status(self.conn, gid, "fixed")
        else:
            for gid in gap_ids:
                mark_gap_status(self.conn, gid, "queued")

        # Always restore to base branch explicitly (do NOT rely on PRManager.original_branch)
        try:
            self._rollback_to_base(base_branch)
        except Exception:
            pass
        
        # Restore the user's original branch best-effort (so running self-improve doesn't yank your context)
        try:
            if user_branch != base_branch:
                self._git(["checkout", user_branch], check=False)
        except Exception:
            pass

        return {
            "goal": computed_goal,
            "baseline": {"score": float(baseline.score()), "gates": list(baseline.gates_failing)},
            "best": {"score": float(best.score()), "gates": list(best.gates_failing)},
            "attempts": attempts,
            "improved": bool(improved_any),
            "pr_url": pr_url,
            "branch": best_branch or base_branch,
        }
        