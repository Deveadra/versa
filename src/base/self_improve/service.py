from __future__ import annotations

import inspect
import re
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from base.self_improve.iteration_controller import (
    IterationBudget,
    LeashPolicy,
    RepoJanitorIterationController,
)
from base.self_improve.models import Proposal, ProposedChange
from base.self_improve.scoreboard import ScoreboardRunner
from base.self_improve.self_improve_db import (
    ensure_self_improve_schema,
    fetch_open_gaps,
    insert_score_run,
    make_gap_fingerprint,
    reconcile_gap_states_for_source,
    upsert_gap,
)
from config.config import settings


@dataclass(frozen=True)
class SelfImproveRunConfig:
    budget: IterationBudget = IterationBudget()
    gap_limit: int = 5
    open_pr: bool = True
    status_callback: Callable[[dict[str, Any]], None] | None = None


class SelfImproveService:
    """
    ONE unified system:
      - Observability: scoreboard -> gaps
      - Continuity: capability_gaps
      - Execution: RepoJanitorIterationController
      - Promotion: PRManager (PRs only, no auto-merge)
      - daily: scoreboard->gaps->iterations->PR
      - manual: optionally includes dream notes as additional context
    """

    def __init__(
        self,
        *,
        repo_root: str | Path,
        db,  # SQLiteConn
        store,  # MemoryStore (optional)
        brain,  # Brain (ProposalEngine uses it)
        code_indexer,
        proposal_engine,
        pr_manager,
    ):
        self.repo = Path(repo_root).resolve()
        self.db = db
        self.conn = db.conn
        self.store = store
        self.brain = brain
        self.code_indexer = code_indexer
        self.proposal_engine = proposal_engine
        self.pr_manager = pr_manager

        self.policy = LeashPolicy(
            allowlist=(
                "src/base/**",
                "src/config/**",
                "scripts/**",
                "tests/**",
                ".github/workflows/**",
                "pyproject.toml",
                "README.md",
                "run.py",
            )
        )

        ensure_self_improve_schema(self.conn)
        self.scoreboard = ScoreboardRunner(self.repo)

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

    def _git_dirty(self) -> bool:
        out = self._git(["status", "--porcelain"], check=False).stdout or ""
        return bool(out.strip())

    def _current_branch(self) -> str:
        return (self._git(["rev-parse", "--abbrev-ref", "HEAD"], check=False).stdout or "").strip()

    def _current_sha(self) -> str:
        return (self._git(["rev-parse", "HEAD"], check=False).stdout or "").strip()

    def _push_branch_compat(self, branch: str) -> None:
        """
        Push branch without assuming PRManager has a specific method name.
        """
        # 1) preferred: pr_manager.push_branch
        fn = getattr(self.pr_manager, "push_branch", None)
        if callable(fn):
            fn(branch)
            return

        # 2) PRManager has .client.push (GitClient)
        client = getattr(self.pr_manager, "client", None)
        if client is not None and hasattr(client, "push"):
            client.push(branch)
            return

        # 3) last resort: raw git
        self._git(["push", "-u", "origin", branch], check=False)

    # -------------------------
    # index helpers
    # -------------------------

    def _index_md(self) -> str:
        """
        Build index markdown robustly regardless of CodeIndexer return shape.
        """
        idx = self.code_indexer.scan(incremental=True)

        # dict style: {"files": {"path": {...}, ...}}
        if isinstance(idx, dict) and isinstance(idx.get("files"), dict):
            lines = ["# Repository Index", ""]
            for path in sorted(idx["files"].keys()):
                lines.append(f"- `{path}`")
            return "\n".join(lines)

        # fallback: CodeIndexer.to_markdown if it can handle whatever idx is
        to_md = getattr(self.code_indexer, "to_markdown", None)
        if callable(to_md):
            try:
                return to_md(idx)
            except Exception:
                pass

        # final fallback: list a sane subset of files
        exts = {".py", ".md", ".toml", ".yml", ".yaml", ".json", ".txt", ".ini"}
        ignore = {
            ".git",
            ".venv",
            "__pycache__",
            ".pytest_cache",
            ".ruff_cache",
            ".mypy_cache",
            "dist",
            "build",
        }
        lines = ["# Repository Index", ""]
        for p in sorted(self.repo.rglob("*")):
            if any(part in ignore for part in p.parts):
                continue
            if p.is_file() and p.suffix.lower() in exts:
                lines.append(f"- `{p.relative_to(self.repo).as_posix()}`")
        return "\n".join(lines)

    # -------------------------
    # db helper compatibility
    # -------------------------

    def _insert_score_run_compat(
        self, *, run_type: str, run: Any, git_branch: str, git_sha: str
    ) -> None:
        """
        Call insert_score_run in a signature-tolerant way.
        Supports multiple historical signatures.
        """
        fn = insert_score_run
        try:
            # Common signature in your current service.py usage
            fn(self.conn, run_type=run_type, run=run, git_branch=git_branch, git_sha=git_sha)
            return
        except TypeError:
            pass

        # Alternative signature: explicit fields
        try:
            mode = getattr(run, "mode", "all")
            fix_enabled = bool(getattr(run, "fix_enabled", False))
            score = float(run.score()) if hasattr(run, "score") else 0.0
            passed = bool(run.passed()) if hasattr(run, "passed") else False
            metrics = run.to_dict() if hasattr(run, "to_dict") else {}
            fn(
                self.conn,
                run_type=run_type,
                mode=mode,
                fix_enabled=fix_enabled,
                git_branch=git_branch,
                git_sha=git_sha,
                score=score,
                passed=passed,
                metrics=metrics,
            )
            return
        except TypeError:
            pass

        # Final: use introspection to build kwargs
        sig = inspect.signature(fn)
        kwargs: dict[str, Any] = {}
        if "run_type" in sig.parameters:
            kwargs["run_type"] = run_type
        if "git_branch" in sig.parameters:
            kwargs["git_branch"] = git_branch
        if "git_sha" in sig.parameters:
            kwargs["git_sha"] = git_sha
        if "run" in sig.parameters:
            kwargs["run"] = run
        if "metrics" in sig.parameters and hasattr(run, "to_dict"):
            kwargs["metrics"] = run.to_dict()
        fn(self.conn, **kwargs)

    # -------------------------
    # misc helpers
    # -------------------------

    def _sanitize_suffix(self, s: str) -> str:
        s = (s or "").lower().strip()
        s = re.sub(r"[^a-z0-9_\-]+", "-", s)
        return (s[:40] or f"run-{int(time.time())}").strip("-")

    def _default_policy(self) -> LeashPolicy:
        return self.policy

    def _changed_files_since_base(self, base: str) -> list[str]:
        out = self._git(["diff", "--name-only", f"{base}..HEAD"], check=False).stdout or ""
        return [ln.strip() for ln in out.splitlines() if ln.strip()]

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

    # -------------------------
    # gap logging
    # -------------------------

    def log_gaps_from_scoreboard(self, run, *, source: str = "scoreboard") -> set[str]:
        active_fingerprints: set[str] = set()

        for name, tr in run.tool_results.items():
            if tr.exit_code == 0:
                continue

            fingerprint = make_gap_fingerprint(
                source, name, str(tr.exit_code), tr.stderr_tail or tr.stdout_tail
            )
            active_fingerprints.add(fingerprint)

            upsert_gap(
                self.conn,
                source=source,
                fingerprint=fingerprint,
                requested_capability=f"pass_{name}_gate",
                observed_failure=(tr.stderr_tail or tr.stdout_tail),
                classification="quality_gate",
                repro_steps=f"Scoreboard tool '{name}' failed (exit={tr.exit_code}).",
                priority=50 if name in ("pytest", "compile") else 25,
                metadata={"tool": name, "exit_code": tr.exit_code},
            )

        reconcile_gap_states_for_source(
            self.conn,
            source=source,
            active_fingerprints=active_fingerprints,
            active_status="queued",
        )

        return active_fingerprints

    def build_goal_from_gaps(self, *, limit: int) -> tuple[str, list[int]]:
        gaps = fetch_open_gaps(self.conn, limit=limit)
        if not gaps:
            return "Reduce failing gates and improve scoreboard reliability.", []

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

    # -------------------------
    # PR creation helpers
    # -------------------------

    def open_pr_from_branch(self, *, base: str, title: str, body: str) -> str:
        changed = self._changed_files_since_base(base)
        changes = [
            ProposedChange(
                path=p,
                apply_mode="replace_block",
                search_anchor=None,
                replacement="(see diff)",
            )
            for p in changed
        ]
        proposal = Proposal(title=title, description=body, changes=changes)
        branch = self._git(["rev-parse", "--abbrev-ref", "HEAD"]).stdout.strip()
        return self.pr_manager.open_pr(branch=branch, proposal=proposal)

    # -------------------------
    # public entrypoints
    # -------------------------

    def run_daily(self, *, cfg: SelfImproveRunConfig = SelfImproveRunConfig()) -> dict[str, Any]:
        return self._run_unified(branch_label="daily-self-improve", cfg=cfg, extra_context=None)

    def run_manual(
        self, *, cfg: SelfImproveRunConfig = SelfImproveRunConfig(), include_dream: bool = True
    ) -> dict[str, Any]:
        dream_context = ""
        if include_dream:
            try:
                # lazy import prevents pulling voice deps unless explicitly requested
                from base.agents.dream import DreamCycle

                dc = DreamCycle()
                notes = dc.run()
                if isinstance(notes, dict):
                    dream_path = dc.write_summary(notes)
                    dream_context = Path(dream_path).read_text(encoding="utf-8")[:4000]
            except Exception as e:
                logger.warning(f"[self-improve] dream cycle failed (non-fatal): {e}")

        result = self._run_unified(
            branch_label="manual-self-improve", cfg=cfg, extra_context=dream_context
        )

        if dream_context.strip():
            fp = make_gap_fingerprint("dream", dream_context[:500])
            upsert_gap(
                self.conn,
                source="dream",
                fingerprint=fp,
                requested_capability="address_dream_cycle_notes",
                observed_failure=dream_context[:2000],
                classification="unknown",
                repro_steps=None,
                priority=10,
                metadata={},
            )
        return result

    def run_interactive_proposal(self, *, instruction: str) -> str:
        """
        Unifies the old propose_code_change pathway behind the service:
          - build index
          - propose patch
          - open PR
          - run tests and update PR body
        """
        base = (settings.github_default_branch or "main").strip()
        index_md = self._index_md()
        proposal = self.proposal_engine.propose(instruction, index_md=index_md)

        suffix = self._sanitize_suffix(proposal.title)
        branch = self.pr_manager.prepare_branch(suffix, base=base)

        results = self.proposal_engine.apply_proposal(proposal)
        failed = [f"- {c.path}: {msg}" for (c, ok, msg) in results if not ok]

        run = self.scoreboard.run(mode="all", fix=False)
        self._insert_score_run_compat(
            run_type="interactive_propose",
            run=run,
            git_branch=branch,
            git_sha=self._current_sha(),
        )
        self.log_gaps_from_scoreboard(run, source="diagnostic")

        if not any(ok for (_, ok, _) in results):
            try:
                self.pr_manager.restore_original_branch()
            except Exception:
                pass
            return "Proposal aborted — no changes could be safely applied."

        self.pr_manager.commit_and_push(branch, proposal.title)
        pr_url = self.pr_manager.open_pr(branch=branch, proposal=proposal)

        try:
            self.pr_manager.run_tests_and_update_pr(branch)
        except Exception:
            pass

        try:
            self.pr_manager.restore_original_branch()
        except Exception:
            pass

        report = []
        if failed:
            report.append("⚠️ Some changes could not be applied:\n" + "\n".join(failed))
        report_text = "\n\n".join(report) if report else "✅ All changes applied successfully."
        return f"Proposal opened: {pr_url}\n\n{report_text}"

    # -------------------------
    # unified engine
    # -------------------------

    def _run_unified(
        self, *, branch_label: str, cfg: SelfImproveRunConfig, extra_context: str | None = None
    ) -> dict[str, Any]:
        base = (settings.github_default_branch or "main").strip()
        suffix = self._sanitize_suffix(f"{branch_label}-{int(time.time())}")
        branch = self.pr_manager.prepare_branch(suffix, base=base, restore_stash=True)

        goal_hint = ""
        if extra_context and extra_context.strip():
            goal_hint = "Dream context:\n" + extra_context.strip()[:4000]

        controller = RepoJanitorIterationController(
            repo_root=str(self.repo),
            db_conn=self.db,
            code_indexer=self.code_indexer,
            proposal_engine=self.proposal_engine,
            policy=self._default_policy(),
            pr_manager=self.pr_manager,
        )

        try:
            result = controller.run(
                goal=goal_hint,
                budget=cfg.budget,
                status_callback=cfg.status_callback,
            )
        except TypeError:
            controller = RepoJanitorIterationController(
                repo_root=str(self.repo),
                db_conn=self.db,
                code_indexer=self.code_indexer,
                proposal_engine=self.proposal_engine,
                pr_manager=None,
                policy=self._default_policy(),
            )
            result = controller.run(
                goal=goal_hint,
                budget=cfg.budget,
                status_callback=cfg.status_callback,
            )

        pr_url = result.get("pr_url")
        promotion_branch = str(result.get("branch") or branch)

        if cfg.open_pr and result.get("improved") and not pr_url:
            self._emit_status(
                cfg.status_callback,
                phase="propose_pr",
                state="start",
                message="Opening improvement PR",
                pct=85,
                branch=promotion_branch,
            )
            try:
                if self._git_dirty():
                    self.pr_manager.commit_and_push(
                        promotion_branch, f"Repo Janitor: {branch_label}"
                    )
                else:
                    self._push_branch_compat(promotion_branch)

                body = (
                    "Repo Janitor unified self-improvement run\n\n"
                    f"### Goal\n{result.get('goal', '')}\n\n"
                    f"### Baseline\nscore={result['baseline']['score']:.2f} gates={result['baseline']['gates']}\n\n"
                    f"### Best\nscore={result['best']['score']:.2f} gates={result['best']['gates']}\n\n"
                    f"Branch: `{promotion_branch}`\n"
                )

                pr_url = self.open_pr_from_branch(
                    base=base,
                    title=f"Repo Janitor: {branch_label}",
                    body=body,
                )

                try:
                    self.pr_manager.run_tests_and_update_pr(promotion_branch)
                except Exception:
                    pass

                self._emit_status(
                    cfg.status_callback,
                    phase="propose_pr",
                    state="complete",
                    message="Improvement PR ready",
                    pct=95,
                    branch=promotion_branch,
                    outcome=pr_url or "branch_pushed",
                )
            except Exception as e:
                self._emit_status(
                    cfg.status_callback,
                    phase="propose_pr",
                    state="error",
                    message="Improvement PR failed",
                    pct=95,
                    branch=promotion_branch,
                    error=str(e),
                )
                logger.exception(f"[self-improve] PR open failed: {e}")

        try:
            self.pr_manager.restore_original_branch()
        except Exception:
            pass

        result["pr_url"] = pr_url
        result["branch"] = result.get("branch") or promotion_branch

        self._record_dream_summary_event(result=result, goal=result.get("goal", ""))

        return result

    def _record_dream_summary_event(self, *, result: dict[str, Any], goal: str) -> None:
        """
        Persist a human-readable summary of the self-improve run into MemoryStore/events.
        """
        try:
            pr_url = result.get("pr_url") or ""
            branch = result.get("branch") or ""
            baseline = result.get("baseline") or {}
            best = result.get("best") or {}
            attempts = result.get("attempts") or []
            improved = bool(result.get("improved"))

            errors = sum(1 for a in attempts if isinstance(a, dict) and a.get("error"))
            rollbacks = sum(
                1
                for a in attempts
                if isinstance(a, dict) and a.get("improved") is False and not a.get("error")
            )

            gaps = fetch_open_gaps(self.conn, limit=3)
            gap_lines = [
                f"- ({g['classification']}) {g['requested_capability']} [prio={g['priority']}]"
                for g in gaps
            ]
            gap_block = "\n".join(gap_lines) if gap_lines else "- none"

            text = (
                "[dream_summary]\n"
                f"Goal:\n{goal}\n\n"
                f"Result: {'IMPROVED' if improved else 'NO CHANGE'}\n"
                f"Branch: {branch}\n"
                f"PR: {pr_url or '(none)'}\n\n"
                # f"Baseline: score={int(baseline.get('score', 0.0)):.2f} gates={baseline.get('gates')}\n"
                # f"Best: score={int(best.get('score', 0.0)):.2f} gates={best.get('gates')}\n"
                f"Baseline: score={float(baseline.get('score', 0.0)):.2f} gates={baseline.get('gates')}\n"
                f"Best: score={float(best.get('score', 0.0)):.2f} gates={best.get('gates')}\n"
                f"Attempts: {len(attempts)} | Rollbacks: {rollbacks} | Errors: {errors}\n\n"
                "Top remaining gaps:\n"
                f"{gap_block}\n"
            )

            if hasattr(self.store, "add_event"):
                self.store.add_event(
                    content=text,
                    importance=0.2,
                    type_="dream_summary",
                    vector_write="sync",
                )
        except Exception as e:
            logger.debug(f"[self-improve] dream summary event skipped: {e}")
