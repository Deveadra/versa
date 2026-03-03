from __future__ import annotations

import json
import re
import subprocess
import time
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

from loguru import logger

from base.self_improve.models import Proposal
from base.self_improve.scoreboard import ScoreboardRunner
from base.self_improve.self_improve_db import insert_attempt, insert_score_run


@dataclass(frozen=True)
class IterationBudget:
    max_seconds: int = 20 * 60
    max_iterations: int = 8
    max_files_changed: int = 20
    max_patch_bytes: int = 256_000


@dataclass(frozen=True)
class LeashPolicy:
    allowlist: tuple[str, ...]
    blocklist: tuple[str, ...] = (
        ".git/**",
        ".venv/**",
        ".aerith/**",
        "**/.env",
        "**/.env.*",
        "**/*id_rsa*",
        "**/*.key",
        "**/*.pem",
        "keys/**",
    )
    allow_full_file_globs: tuple[str, ...] = (
        "tests/**",
        ".github/workflows/**",
        "scripts/**",
        "README.md",
        "pyproject.toml",
        "run.py",
    )
    secret_patterns: tuple[re.Pattern[str], ...] = (
        re.compile(r"-----BEGIN (RSA|EC|OPENSSH)? ?PRIVATE KEY-----"),
        re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),
        re.compile(r"\bOPENAI_API_KEY\b"),
    )


class LeashViolation(RuntimeError):
    pass


class RepoJanitorIterationController:
    """
    Tight loop executor. Assumes caller already checked out a working branch.

    Uses:
      - CodeIndexer.scan() + CodeIndexer.to_markdown()
      - ProposalEngine.propose(instruction, index_md)
      - ProposalEngine.apply_proposal(proposal)
      - ScoreboardRunner.run()
      - git restore/clean rollback
      - persists repo_score_runs + repo_improvement_attempts each iteration
    """

    def __init__(
        self,
        *,
        repo_root: str | Path,
        db_conn,  # SQLiteConn or sqlite3.Connection
        code_indexer,
        proposal_engine,
        policy: LeashPolicy,
    ):
        self.repo = Path(repo_root).resolve()
        self.db = getattr(db_conn, "conn", db_conn)
        self.code_indexer = code_indexer
        self.proposal_engine = proposal_engine
        self.policy = policy
        self.scoreboard = ScoreboardRunner(self.repo)

        self._aerith_dir = self.repo / ".aerith"
        self._aerith_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # git helpers
    # -------------------------

    def _git(self, args: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
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

    def _git_branch(self) -> str:
        return self._git(["rev-parse", "--abbrev-ref", "HEAD"]).stdout.strip()

    def _git_sha(self) -> str:
        return self._git(["rev-parse", "HEAD"]).stdout.strip()

    def _git_changed_paths(self) -> list[str]:
        out = self._git(["status", "--porcelain"], check=False).stdout or ""
        paths: list[str] = []
        for line in out.splitlines():
            if not line.strip():
                continue
            paths.append(line[3:].strip())
        return paths

    def _git_restore_to(self, sha: str) -> None:
        self._git(["restore", "--source", sha, "--staged", "--worktree", "."])
        self._git(["clean", "-fd"])

    # -------------------------
    # leash helpers
    # -------------------------

    def _is_allowed(self, relpath: str) -> bool:
        rp = relpath.replace("\\", "/")
        if any(fnmatch(rp, pat) for pat in self.policy.blocklist):
            return False
        return any(fnmatch(rp, pat) for pat in self.policy.allowlist)

    def _scan_secrets(self, s: str) -> None:
        for pat in self.policy.secret_patterns:
            if pat.search(s or ""):
                raise LeashViolation(f"secret-like content detected: {pat.pattern}")

    def _proposal_bytes(self, proposal: Proposal) -> int:
        total = 0
        for ch in proposal.changes:
            total += len((ch.replacement or "").encode("utf-8", errors="ignore"))
            total += len((ch.search_anchor or "").encode("utf-8", errors="ignore"))
        return total

    def _leash_check_proposal(self, proposal: Proposal, budget: IterationBudget) -> None:
        if not proposal.changes:
            raise LeashViolation("proposal has no changes")

        if len(proposal.changes) > budget.max_files_changed:
            raise LeashViolation(f"proposal exceeds max_files_changed: {len(proposal.changes)}")

        total_bytes = self._proposal_bytes(proposal)
        if total_bytes > budget.max_patch_bytes:
            raise LeashViolation(f"proposal exceeds max_patch_bytes: {total_bytes}")

        for ch in proposal.changes:
            rp = str(ch.path).replace("\\", "/")
            if not self._is_allowed(rp):
                raise LeashViolation(f"path not allowlisted: {rp}")

            if ch.apply_mode == "full_file":
                if not any(fnmatch(rp, pat) for pat in self.policy.allow_full_file_globs):
                    raise LeashViolation(f"full_file not allowed for path: {rp}")

            self._scan_secrets(ch.replacement or "")

    # -------------------------
    # index
    # -------------------------

    def _write_index_md(self) -> str:
        idx = self.code_indexer.scan(incremental=True)
        md = self.code_indexer.to_markdown(idx)
        (self._aerith_dir / "index.md").write_text(md, encoding="utf-8")
        return md

    # -------------------------
    # scoring preference
    # -------------------------

    def _better(self, after, before) -> bool:
        if after.gates_failing != before.gates_failing:
            return after.gates_failing < before.gates_failing
        return after.score() > before.score()

    # -------------------------
    # run loop
    # -------------------------

    def run(self, *, goal: str, budget: IterationBudget) -> dict[str, Any]:
        started = time.time()
        branch = self._git_branch()
        baseline_sha = self._git_sha()

        baseline = self.scoreboard.run(mode="all", fix=False)
        baseline_id = insert_score_run(
            self.db,
            run_type="baseline",
            run=baseline,
            git_branch=branch,
            git_sha=baseline_sha,
        )

        best_sha = baseline_sha
        best_run = baseline
        best_run_id = baseline_id

        attempts: list[dict[str, Any]] = []

        for i in range(budget.max_iterations):
            if (time.time() - started) > budget.max_seconds:
                logger.info("[repo-janitor] time budget hit; stopping")
                break

            if best_run.passed():
                logger.info("[repo-janitor] all gates passing; stopping")
                break

            before_id = insert_score_run(
                self.db,
                run_type="iteration_before",
                run=best_run,
                git_branch=branch,
                git_sha=best_sha,
            )

            index_md = self._write_index_md()

            instruction = (
                "You are Repo Janitor operating inside THIS repo.\n"
                f"GOAL: {goal}\n\n"
                "Rules:\n"
                "- Make the smallest correct change.\n"
                "- Use replace_block with an anchor for existing code files.\n"
                "- Do not write secrets.\n"
                "- Stay inside allowlist.\n\n"
                f"Current scoreboard: gates_failing={best_run.gates_failing}, score={best_run.score():.2f}\n\n"
                "Repository index:\n"
                f"{index_md}\n"
            )

            proposal: Proposal | None = None
            proposal_json: str | None = None

            try:
                proposal = self.proposal_engine.propose(instruction, index_md=index_md)
                proposal_json = json.dumps(
                    proposal.model_dump() if hasattr(proposal, "model_dump") else proposal.dict(),
                    ensure_ascii=False,
                )

                self._leash_check_proposal(proposal, budget)

                apply_results = self.proposal_engine.apply_proposal(proposal)
                ok_any = any(ok for (_, ok, _) in apply_results)
                if not ok_any:
                    raise RuntimeError("proposal apply produced no successful changes")

                changed = self._git_changed_paths()
                if not changed:
                    raise RuntimeError("apply reported changes, but git shows none")

                # post-apply leash check (changed files only)
                if len(changed) > budget.max_files_changed:
                    raise LeashViolation(f"changed files exceed budget: {len(changed)}")
                for p in changed:
                    if not self._is_allowed(p):
                        raise LeashViolation(f"post-apply changed file not allowlisted: {p}")

                after = self.scoreboard.run(mode="all", fix=False)
                after_id = insert_score_run(
                    self.db,
                    run_type="iteration_after",
                    run=after,
                    git_branch=branch,
                    git_sha=self._git_sha(),
                )

                improved = self._better(after, best_run)

                insert_attempt(
                    self.db,
                    iteration=i,
                    baseline_run_id=baseline_id,
                    before_run_id=before_id,
                    after_run_id=after_id,
                    branch=branch,
                    proposal_title=(proposal.title if proposal else None),
                    proposal_json=proposal_json,
                    improved=improved,
                )

                attempts.append(
                    {
                        "iter": i,
                        "improved": improved,
                        "gates_before": best_run.gates_failing,
                        "gates_after": after.gates_failing,
                        "score_before": best_run.score(),
                        "score_after": after.score(),
                        "changed_files": changed,
                        "title": proposal.title if proposal else None,
                    }
                )

                if improved:
                    self._git(["add", "-A"])
                    self._git(
                        ["commit", "-m", f"Repo Janitor iteration {i}: improve scoreboard"],
                        check=False,
                    )
                    best_sha = self._git_sha()
                    best_run = after
                    best_run_id = after_id
                else:
                    self._git_restore_to(best_sha)

            except Exception as e:
                logger.exception(f"[repo-janitor] iter={i} failed: {e}")
                self._git_restore_to(best_sha)

                insert_attempt(
                    self.db,
                    iteration=i,
                    baseline_run_id=baseline_id,
                    before_run_id=before_id,
                    after_run_id=None,
                    branch=branch,
                    proposal_title=(proposal.title if proposal else None) if proposal else None,
                    proposal_json=proposal_json,
                    improved=False,
                    error_text=str(e),
                )

                attempts.append({"iter": i, "improved": False, "error": str(e)})

        return {
            "branch": branch,
            "baseline": {"run_id": baseline_id, "score": baseline.score(), "gates": baseline.gates_failing},
            "best": {"run_id": best_run_id, "score": best_run.score(), "gates": best_run.gates_failing},
            "improved": self._better(best_run, baseline),
            "attempts": attempts,
            "best_sha": best_sha,
        }