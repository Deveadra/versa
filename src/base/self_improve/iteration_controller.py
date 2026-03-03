# src/base/self_improve/score_types.py

from __future__ import annotations

import json
import os
import re
import subprocess
import sqlite3
import time
from dataclasses import asdict, dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

from loguru import logger


from base.database.sqlite import SQLiteConn
from base.self_improve.code_indexer import CodeIndexer
from base.self_improve.pr_manager import PRManager
from base.self_improve.proposal_engine import ProposalEngine
from base.self_improve.scoreboard import ScoreboardRunner
from base.self_improve.score_types import ScoreboardRun


# -------------------------
# Budget + policy
# -------------------------

@dataclass(frozen=True)
class IterationBudget:
    max_seconds: int = 20 * 60
    max_iterations: int = 8
    max_files_changed: int = 20
    max_patch_bytes: int = 256_000
    open_pr_on_success: bool = True


@dataclass(frozen=True)
class LeashPolicy:
    # Glob patterns, repo-root relative (forward slashes)
    allowlist: tuple[str, ...]
    blocklist: tuple[str, ...] = (
        ".git/**",
        ".venv/**",
        "**/.env",
        "**/.env.*",
        "**/*id_rsa*",
        "**/*_rsa",
        "**/*.key",
        "**/*.pem",
        "keys/**",
    )

    # Simple secret heuristics
    secret_patterns: tuple[re.Pattern[str], ...] = (
        re.compile(r"-----BEGIN (RSA|EC|OPENSSH)? ?PRIVATE KEY-----"),
        re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),          # common API key prefix
        re.compile(r"\bOPENAI_API_KEY\b"),
        re.compile(r"\bGITHUB_TOKEN\b\s*="),
    )

    # Restrict high-risk rewrite behavior
    allow_full_file_globs: tuple[str, ...] = (
        "tests/**",
        ".github/workflows/**",
        "scripts/**",
        "README.md",
        "pyproject.toml",
    )


class LeashViolation(RuntimeError):
    pass


# -------------------------
# Git helpers (robust rollback)
# -------------------------

class GitRepo:
    def __init__(self, repo_root: Path):
        self.repo = repo_root

    def _run(self, args: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(self.repo),
            text=True,
            capture_output=True,
        )
        if check and proc.returncode != 0:
            raise RuntimeError(
                f"git {' '.join(args)} failed\n"
                f"stdout:\n{proc.stdout}\n"
                f"stderr:\n{proc.stderr}\n"
            )
        return proc

    def current_branch(self) -> str:
        return self._run(["rev-parse", "--abbrev-ref", "HEAD"]).stdout.strip()

    def head_sha(self) -> str:
        return self._run(["rev-parse", "HEAD"]).stdout.strip()

    def checkout_new_branch(self, branch: str) -> None:
        self._run(["checkout", "-b", branch])

    def checkout(self, ref: str) -> None:
        self._run(["checkout", ref])

    def restore_to(self, sha: str) -> None:
        # Restore tracked files
        self._run(["restore", "--source", sha, "--staged", "--worktree", "."])
        # Remove untracked files/dirs created by a proposal
        self._run(["clean", "-fd"])

    def diff_name_only(self) -> list[str]:
        out = self._run(["diff", "--name-only"]).stdout.strip()
        return [line.strip() for line in out.splitlines() if line.strip()]

    def add_all(self) -> None:
        self._run(["add", "-A"])

    def commit(self, message: str) -> None:
        # allow empty commits? no — empty means proposal did nothing
        self._run(["commit", "-m", message], check=False)

    def has_staged_changes(self) -> bool:
        proc = self._run(["diff", "--cached", "--name-only"], check=False)
        return bool(proc.stdout.strip())

    def push(self, branch: str, remote: str = "origin") -> None:
        self._run(["push", "-u", remote, branch])

    def delete_branch(self, branch: str) -> None:
        self._run(["branch", "-D", branch], check=False)


# -------------------------
# DB persistence helpers
# -------------------------

class RepoJanitorDB:
    """
    Persists:
      - repo_score_runs
      - repo_improvement_attempts
    Expects migrations to have created these tables.
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def insert_score_run(
        self,
        *,
        run_type: str,
        run: ScoreboardRun,
        git_branch: str,
        git_sha: str,
    ) -> int:
        payload = {
            "mode": run.mode,
            "fix_enabled": bool(run.fix_enabled),
            "total_duration_ms": run.total_duration_ms,
            "gates_failing": run.gates_failing,
            "score": run.score(),
            "tool_results": {
                k: {
                    "name": v.name,
                    "exit_code": v.exit_code,
                    "duration_ms": v.duration_ms,
                    "stdout_tail": v.stdout_tail,
                    "stderr_tail": v.stderr_tail,
                    "parsed": v.parsed,
                }
                for k, v in run.tool_results.items()
            },
        }

        with self._conn() as conn:
            cur = conn.execute(
                """
                INSERT INTO repo_score_runs(
                  run_type, mode, fix_enabled, git_branch, git_sha, score, passed, metrics_json
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_type,
                    run.mode,
                    1 if run.fix_enabled else 0,
                    git_branch,
                    git_sha,
                    float(run.score()),
                    1 if run.passed() else 0,
                    json.dumps(payload, ensure_ascii=False),
                ),
            )
            return int(cur.lastrowid)

    def insert_attempt(
        self,
        *,
        iteration: int,
        baseline_run_id: int,
        before_run_id: int,
        after_run_id: int | None,
        branch: str,
        proposal_title: str | None,
        proposal_json: str | None,
        improved: bool,
        pr_url: str | None = None,
        error_text: str | None = None,
    ) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                """
                INSERT INTO repo_improvement_attempts(
                  iteration, baseline_run_id, before_run_id, after_run_id,
                  proposal_title, proposal_json, branch, pr_url,
                  improved, error_text
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    iteration,
                    baseline_run_id,
                    before_run_id,
                    after_run_id,
                    proposal_title,
                    proposal_json,
                    branch,
                    pr_url,
                    1 if improved else 0,
                    error_text,
                ),
            )
            return int(cur.lastrowid)


# -------------------------
# Controller
# -------------------------

class RepoJanitorIterationController:
    """
    Loop: baseline -> (index -> propose -> leash -> apply -> verify -> keep/rollback) * N -> optional PR
    """

    def __init__(
        self,
        *,
        repo_root: str | Path,
        db_path: str | Path,
        pr_manager: Any,
        policy: LeashPolicy,
    ):
        self.repo = Path(repo_root).resolve()
        self.git = GitRepo(self.repo)
        self.db = RepoJanitorDB(Path(db_path))
        self.scoreboard = ScoreboardRunner(self.repo)
        self.indexer = CodeIndexer(repo_root=str(self.repo))
        self.proposer = ProposalEngine(repo_root=str(self.repo))
        self.pr_manager = pr_manager
        self.policy = policy

        self._aerith_dir = self.repo / ".aerith"
        self._aerith_dir.mkdir(parents=True, exist_ok=True)

    def _rel(self, p: Path) -> str:
        rel = p.as_posix()
        return rel[2:] if rel.startswith("./") else rel

    def _is_allowed_path(self, relpath: str) -> bool:
        rp = relpath.replace("\\", "/")
        if any(fnmatch(rp, pat) for pat in self.policy.blocklist):
            return False
        return any(fnmatch(rp, pat) for pat in self.policy.allowlist)

    def _scan_for_secrets(self, text: str) -> None:
        for pat in self.policy.secret_patterns:
            if pat.search(text or ""):
                raise LeashViolation(f"secret-like content detected by pattern: {pat.pattern}")

    def _proposal_to_json(self, proposal: Any) -> str:
        # Pydantic v2
        if hasattr(proposal, "model_dump"):
            return json.dumps(proposal.model_dump(), ensure_ascii=False)
        # Pydantic v1
        if hasattr(proposal, "dict"):
            return json.dumps(proposal.dict(), ensure_ascii=False)
        # Dataclass-like
        if hasattr(proposal, "__dict__"):
            return json.dumps(proposal.__dict__, ensure_ascii=False)
        return json.dumps({"repr": repr(proposal)}, ensure_ascii=False)

    def _build_index_md(self) -> str:
        """
        Uses CodeIndexer. Writes `.aerith/index.md` every iteration (so proposals can cite it).
        """
        index_md_path = self._aerith_dir / "index.md"
        try:
            # expected shape (based on prior design): index = indexer.build_index(); md = indexer.to_markdown(index)
            idx = self.indexer.build_index()
            md = self.indexer.to_markdown(idx)
        except Exception as e:
            logger.warning(f"[repo-janitor] CodeIndexer failed; continuing without index_md: {e}")
            md = ""
        index_md_path.write_text(md, encoding="utf-8")
        return md

    def _leash_check_proposal(self, proposal: Any) -> None:
        """
        Validate proposed paths + sizes BEFORE applying.
        Assumes proposal has `.changes` where each change has `.path` and `.change_type`.
        """
        changes = getattr(proposal, "changes", None)
        if not changes:
            raise LeashViolation("proposal has no changes")

        if len(changes) >  self.policy and 0:  # placeholder guard removed below
            pass

        if len(changes) > 20:
            raise LeashViolation(f"proposal changes exceed max_files_changed: {len(changes)}")

        total_bytes = 0
        for ch in changes:
            path = getattr(ch, "path", None)
            if not path:
                raise LeashViolation("proposal change missing path")

            rp = str(path).replace("\\", "/")
            if not self._is_allowed_path(rp):
                raise LeashViolation(f"path not allowlisted: {rp}")

            change_type = getattr(ch, "change_type", "") or getattr(ch, "type", "")
            if str(change_type) == "full_file":
                # full-file writes only allowed in a limited set of safe globs
                if not any(fnmatch(rp, pat) for pat in self.policy.allow_full_file_globs):
                    raise LeashViolation(f"full_file not allowed for path: {rp}")

            # bytes estimate (best-effort)
            content = getattr(ch, "new_content", None) or getattr(ch, "content", None) or ""
            total_bytes += len(content.encode("utf-8", errors="ignore"))
            self._scan_for_secrets(content)

        if total_bytes > 256_000:
            raise LeashViolation(f"proposal patch bytes exceed max_patch_bytes: {total_bytes}")

    def _better(self, a: ScoreboardRun, b: ScoreboardRun) -> bool:
        """
        Primary objective: fewer failing gates.
        Secondary: fewer pytest failures, fewer ruff violations, faster total runtime.
        """
        def key(run: ScoreboardRun) -> tuple[int, int, int, float]:
            pytest_fail = int(run.tool_results.get("pytest").parsed.get("failures", 0)) if run.tool_results.get("pytest") else 0
            ruff_cnt = int(run.tool_results.get("ruff").parsed.get("count", 0)) if run.tool_results.get("ruff") else 0
            return (run.gates_failing, pytest_fail, ruff_cnt, float(run.total_duration_ms))

        return key(a) < key(b)

    def run(self, *, goal: str, budget: IterationBudget) -> dict[str, Any]:
        start = time.time()

        original_branch = self.git.current_branch()
        work_branch = f"aerith/repo-janitor/{int(start)}"
        self.git.checkout_new_branch(work_branch)

        best_sha = self.git.head_sha()

        # Baseline run (and persist)
        baseline = self.scoreboard.run(mode="all", fix=False)
        baseline_id = self.db.insert_score_run(
            run_type="baseline",
            run=baseline,
            git_branch=work_branch,
            git_sha=best_sha,
        )
        best_run = baseline
        best_run_id = baseline_id

        attempts: list[dict[str, Any]] = []

        for i in range(budget.max_iterations):
            if (time.time() - start) > budget.max_seconds:
                logger.info("[repo-janitor] budget max_seconds hit; stopping")
                break

            if best_run.passed():
                logger.info("[repo-janitor] all gates passing; stopping")
                break

            # "before" run snapshot for this iteration
            before_id = self.db.insert_score_run(
                run_type="iteration_before",
                run=best_run,
                git_branch=work_branch,
                git_sha=best_sha,
            )

            index_md = self._build_index_md()

            instruction = (
                "You are Repo Janitor operating INSIDE this repository.\n"
                f"GOAL: {goal}\n\n"
                "Hard rules:\n"
                "- Make the smallest correct change.\n"
                "- Stay inside the allowlist.\n"
                "- Do not touch secrets/credentials.\n"
                "- Prefer replace_block edits with anchors.\n"
                "- If tests fail, fix tests before refactors.\n\n"
                "Scoreboard context:\n"
                f"- gates_failing={best_run.gates_failing}\n"
                f"- ruff_count={best_run.tool_results.get('ruff').parsed.get('count', 0) if best_run.tool_results.get('ruff') else 0}\n"
                f"- black_exit={best_run.tool_results.get('black').exit_code if best_run.tool_results.get('black') else 'NA'}\n"
                f"- pytest_failures={best_run.tool_results.get('pytest').parsed.get('failures', 0) if best_run.tool_results.get('pytest') else 0}\n\n"
                "Repo map (index_md):\n"
                f"{index_md}\n"
            )

            proposal = None
            proposal_json = None
            proposal_title = None

            try:
                proposal = self.proposer.propose(instruction=instruction, index_md=index_md)
                proposal_title = getattr(proposal, "title", None)
                proposal_json = self._proposal_to_json(proposal)

                # leash before apply
                self._leash_check_proposal(proposal)

                # apply
                self.proposer.apply(proposal)

                # leash after apply (validate actual diff paths)
                changed = self.git.diff_name_only()
                if len(changed) > budget.max_files_changed:
                    raise LeashViolation(f"git diff files exceed max_files_changed: {len(changed)}")

                for f in changed:
                    rp = f.replace("\\", "/")
                    if not self._is_allowed_path(rp):
                        raise LeashViolation(f"post-apply changed file not allowlisted: {rp}")

                # verify
                after = self.scoreboard.run(mode="all", fix=False)
                improved = self._better(after, best_run)

                after_id = self.db.insert_score_run(
                    run_type="iteration_after",
                    run=after,
                    git_branch=work_branch,
                    git_sha=self.git.head_sha(),
                )

                if improved:
                    # Keep: commit and update best_sha/best_run
                    self.git.add_all()
                    if not self.git.has_staged_changes():
                        # proposal claimed changes but git sees nothing staged
                        raise RuntimeError("proposal produced no staged changes")

                    self.git.commit(f"Repo Janitor iteration {i}: improve scoreboard")
                    best_sha = self.git.head_sha()
                    best_run = after
                    best_run_id = after_id
                    logger.info(f"[repo-janitor] iter={i} improved score={best_run.score():.2f} gates={best_run.gates_failing}")
                else:
                    # Rollback to best known good SHA
                    logger.info(f"[repo-janitor] iter={i} no improvement; rollback to {best_sha[:8]}")
                    self.git.restore_to(best_sha)

                self.db.insert_attempt(
                    iteration=i,
                    baseline_run_id=baseline_id,
                    before_run_id=before_id,
                    after_run_id=after_id,
                    branch=work_branch,
                    proposal_title=proposal_title,
                    proposal_json=proposal_json,
                    improved=improved,
                )

                attempts.append(
                    {
                        "iter": i,
                        "improved": improved,
                        "score_before": float(best_run.score()) if not improved else float(after.score()),
                        "score_after": float(after.score()),
                        "proposal_title": proposal_title,
                    }
                )

                if improved and best_run.passed():
                    break

            except Exception as e:
                logger.exception(f"[repo-janitor] iter={i} failed: {e}")
                # hard rollback
                self.git.restore_to(best_sha)

                # attempt row with error
                self.db.insert_attempt(
                    iteration=i,
                    baseline_run_id=baseline_id,
                    before_run_id=before_id,
                    after_run_id=None,
                    branch=work_branch,
                    proposal_title=proposal_title,
                    proposal_json=proposal_json,
                    improved=False,
                    error_text=str(e),
                )

                attempts.append({"iter": i, "improved": False, "error": str(e)})

        pr_url = None
        if budget.open_pr_on_success and self._better(best_run, baseline):
            # push + open PR (review gate stays with YOU)
            try:
                self.git.push(work_branch)

                # Your PRManager likely wraps GitHub API.
                # Expected interface: open_pr(branch, title, body) OR similar.
                body = (
                    f"Repo Janitor self-improvement run\n\n"
                    f"Baseline score: {baseline.score():.2f} (gates={baseline.gates_failing})\n"
                    f"Best score: {best_run.score():.2f} (gates={best_run.gates_failing})\n\n"
                    f"Work branch: `{work_branch}`\n"
                    f"Baseline run id: {baseline_id}\n"
                    f"Best run id: {best_run_id}\n"
                )

                pr_url = self.pr_manager.open_pr(
                    branch=work_branch,
                    title=f"Repo Janitor: {goal}",
                    body=body,
                )
                logger.info(f"[repo-janitor] opened PR: {pr_url}")
            except Exception as e:
                logger.exception(f"[repo-janitor] PR open failed: {e}")

        # return to original branch (no destructive resets)
        self.git.checkout(original_branch)

        return {
            "baseline": {"score": float(baseline.score()), "gates_failing": baseline.gates_failing, "run_id": baseline_id},
            "best": {"score": float(best_run.score()), "gates_failing": best_run.gates_failing, "run_id": best_run_id},
            "branch": work_branch,
            "attempts": attempts,
            "pr_url": pr_url,
        }
        
        
class IterationBudget:
    def __init__(self, *, max_seconds: int = 1200, max_iters: int = 8):
        self.max_seconds = int(max_seconds)
        self.max_iters = int(max_iters)


class RepoJanitorController:
    def __init__(self, repo_root: str | Path):
        self.repo = Path(repo_root).resolve()
        self.scoreboard = ScoreboardRunner(self.repo)
        self.proposals = ProposalEngine(str(self.repo))
        self.pr = PRManager(str(self.repo))
        self.db = SQLiteConn(str(self.repo / "aerith.db"))  # or settings.db_path if preferred

    def _better(self, a: ScoreboardRun, b: ScoreboardRun) -> bool:
        # a is better than b?
        a_key = (a.gates_failing, a.tool_results["pytest"].parsed.get("failures", 0), a.tool_results["ruff"].parsed.get("count", 0), a.total_duration_ms)
        b_key = (b.gates_failing, b.tool_results["pytest"].parsed.get("failures", 0), b.tool_results["ruff"].parsed.get("count", 0), b.total_duration_ms)
        return a_key < b_key

    def run(self, *, goal: str, budget: IterationBudget, open_pr: bool = False) -> dict:
        t0 = time.time()
        baseline = self.scoreboard.run(mode="all", fix=False)
        best = baseline

        attempts = []
        for i in range(budget.max_iters):
            if time.time() - t0 > budget.max_seconds:
                break
            if best.passed():
                break

            instruction = (
                "You are Repo Janitor inside this repository.\n"
                "Goal: improve repository hygiene to maximize the scoreboard.\n"
                "Constraints:\n"
                "- Make the smallest correct change.\n"
                "- Prefer adding/adjusting tests and configuration.\n"
                "- Do not change secrets or credentials handling.\n"
                "- Do not rewrite entire files unless absolutely necessary.\n\n"
                "Current failures (summary):\n"
                f"- gates_failing={best.gates_failing}\n"
                f"- ruff_count={best.tool_results['ruff'].parsed.get('count', 0)}\n"
                f"- pytest_failures={best.tool_results['pytest'].parsed.get('failures', 0)}\n"
                f"- black_exit={best.tool_results['black'].exit_code}\n"
            )

            proposal = self.proposals.propose(instruction, index_md="(repo map omitted in v1)")
            branch = self.pr.prepare_branch(f"repo-janitor-{int(time.time())}-{i}")
            applied = self.proposals.apply_proposal(proposal)

            after = self.scoreboard.run(mode="all", fix=False)
            improved = self._better(after, best)

            if improved:
                best = after
                logger.info(f"[iterate] improved on iter {i}: score={best.score():.2f}")
            else:
                # rollback: restore original branch and working tree
                logger.info(f"[iterate] no improvement on iter {i}; rolling back")
                self.pr.restore_original_branch()

            attempts.append(
                {
                    "iter": i,
                    "branch": branch,
                    "improved": improved,
                    "baseline_score": baseline.score(),
                    "after_score": after.score(),
                    "proposal_title": proposal.title,
                }
            )

            if improved and best.passed():
                break

        result = {
            "baseline": {"score": baseline.score(), "gates_failing": baseline.gates_failing},
            "best": {"score": best.score(), "gates_failing": best.gates_failing},
            "attempts": attempts,
        }

        # Optional promotion
        if open_pr and self._better(best, baseline):
            try:
                self.pr.commit_and_push(self.pr.current_branch, f"Repo Janitor: {goal}")
                pr_url = self.pr.open_pr(self.pr.current_branch, proposal=None)  # adapt to your PRManager signature
                result["pr_url"] = pr_url
            except Exception as e:
                result["promotion_error"] = str(e)

        return result
