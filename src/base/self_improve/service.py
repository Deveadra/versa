from __future__ import annotations

import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from base.agents.dream import DreamCycle
from base.self_improve.iteration_controller import IterationBudget, LeashPolicy, RepoJanitorIterationController
from base.self_improve.models import Proposal, ProposedChange
from base.self_improve.scoreboard import ScoreboardRunner
from base.self_improve.self_improve_db import (
    ensure_self_improve_schema,
    fetch_open_gaps,
    insert_score_run,
    make_gap_fingerprint,
    mark_gap_status,
    upsert_gap,
)
from config.config import settings


@dataclass(frozen=True)
class SelfImproveRunConfig:
    budget: IterationBudget = IterationBudget()
    gap_limit: int = 5
    open_pr: bool = True


class SelfImproveService:
    """
    ONE unified system:
      - Observability: scoreboard -> gaps
      - Continuity: capability_gaps
      - Execution: RepoJanitorIterationController
      - Promotion: PRManager (PRs only, no auto-merge)
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

        ensure_self_improve_schema(self.conn)
        self.scoreboard = ScoreboardRunner(self.repo)

    # -------------------------
    # helpers
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

    def _sanitize_suffix(self, s: str) -> str:
        s = (s or "").lower().strip()
        s = re.sub(r"[^a-z0-9_\-]+", "-", s)
        return (s[:40] or f"run-{int(time.time())}").strip("-")

    def _default_policy(self) -> LeashPolicy:
        # NOTE: we include .aerith/** in controller blocklist; it is internal only.
        return LeashPolicy(
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

    def _changed_files_since_base(self, base: str) -> list[str]:
        # base is expected to exist locally (PRManager prepares from base)
        out = self._git(["diff", "--name-only", f"{base}..HEAD"], check=False).stdout or ""
        return [ln.strip() for ln in out.splitlines() if ln.strip()]

    # -------------------------
    # gap logging
    # -------------------------

    def log_gaps_from_scoreboard(self, run, *, source: str = "scoreboard") -> None:
        for name, tr in run.tool_results.items():
            if tr.exit_code == 0:
                continue
            fingerprint = make_gap_fingerprint(source, name, str(tr.exit_code), tr.stderr_tail or tr.stdout_tail)
            upsert_gap(
                self.conn,
                source=source,
                fingerprint=fingerprint,
                requested_capability=f"pass_{name}_gate",
                observed_failure=(tr.stderr_tail or tr.stdout_tail),
                classification="quality_gate",
                repro_steps=f"python -m {name} (see scoreboard)",
                priority=50 if name in ("pytest", "compile") else 25,
                metadata={"tool": name, "exit_code": tr.exit_code},
            )

    def build_goal_from_gaps(self, *, limit: int) -> tuple[str, list[int]]:
        gaps = fetch_open_gaps(self.conn, limit=limit)
        if not gaps:
            return "Reduce failing gates and improve scoreboard reliability.", []

        lines = ["Fix the highest priority open capability gaps:"]
        ids: list[int] = []
        for g in gaps:
            ids.append(int(g["id"]))
            lines.append(f"- [{g['classification']}] {g['requested_capability']} (priority={g['priority']})")
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
        return self._run_unified(branch_label="daily-self-improve", cfg=cfg, include_dream=False)

    def run_manual(self, *, cfg: SelfImproveRunConfig = SelfImproveRunConfig(), include_dream: bool = True) -> dict[str, Any]:
        dream_context = ""
        if include_dream:
            try:
                dc = DreamCycle()
                notes = dc.run()
                if isinstance(notes, dict):
                    dream_path = dc.write_summary(notes)
                    dream_context = Path(dream_path).read_text(encoding="utf-8")[:4000]
            except Exception as e:
                logger.warning(f"[self-improve] dream cycle failed (non-fatal): {e}")

        result = self._run_unified(branch_label="manual-self-improve", cfg=cfg, include_dream=False)
        if dream_context.strip():
            # log dream context as a gap-like signal (doesn't spam duplicates)
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
        repo_root = str(self.repo)
        index = self.code_indexer.scan(incremental=True)
        index_md = self.code_indexer.to_markdown(index)

        proposal = self.proposal_engine.propose(instruction, index_md=index_md)

        suffix = self._sanitize_suffix(proposal.title)
        branch = self.pr_manager.prepare_branch(suffix, base=settings.github_default_branch)

        results = self.proposal_engine.apply_proposal(proposal)
        failed = [f"- {c.path}: {msg}" for (c, ok, msg) in results if not ok]

        # scoreboard snapshot for PR context
        run = self.scoreboard.run(mode="all", fix=False)
        insert_score_run(
            self.conn,
            run_type="interactive_propose",
            run=run,
            git_branch=branch,
            git_sha=self._git(["rev-parse", "HEAD"]).stdout.strip(),
        )
        self.log_gaps_from_scoreboard(run, source="diagnostic")

        if not any(ok for (_, ok, _) in results):
            try:
                self.pr_manager.restore_original_branch()
            except Exception:
                pass
            return "Proposal aborted — no changes could be safely applied."

        # commit + push + PR
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

    def run_diagnostic_autofix_pr(
        self,
        *,
        mode: str,
        diag_output: str,
        issues_preview: str,
        benchmarks_preview: str,
    ) -> str | None:
        """
        Called when run_diagnostic(fix=True) already modified the working tree.
        Opens a PR using the same PRManager plumbing.
        """
        # If nothing changed, nothing to PR
        dirty = self._git(["status", "--porcelain"], check=False).stdout.strip()
        if not dirty:
            return None

        suffix = self._sanitize_suffix(f"diag-autofix-{int(time.time())}")
        branch = self.pr_manager.prepare_branch(suffix, base=settings.github_default_branch)

        self.pr_manager.commit_and_push(branch, "chore(diagnostics): auto-fix & hygiene")

        body = (
            "Automated diagnostics completed.\n\n"
            f"**Mode:** {mode}\n\n"
            f"### Benchmarks\n{benchmarks_preview or 'n/a'}\n\n"
            f"### Issues (preview)\n{issues_preview or 'n/a'}\n\n"
            "### Tool Output (tail)\n"
            f"```\n{(diag_output or '')[-2000:]}\n```"
        )

        pr_url = self.open_pr_from_branch(
            base=settings.github_default_branch,
            title="chore(diagnostics): auto-fix & hygiene",
            body=body,
        )

        try:
            self.pr_manager.run_tests_and_update_pr(branch)
        except Exception:
            pass

        try:
            self.pr_manager.restore_original_branch()
        except Exception:
            pass

        return pr_url

    # -------------------------
    # unified engine
    # -------------------------

    def _run_unified(self, *, branch_label: str, cfg: SelfImproveRunConfig, include_dream: bool) -> dict[str, Any]:
        base = settings.github_default_branch
        suffix = self._sanitize_suffix(f"{branch_label}-{int(time.time())}")
        branch = self.pr_manager.prepare_branch(suffix, base=base)

        # baseline scoreboard => gaps
        baseline = self.scoreboard.run(mode="all", fix=False)
        self.log_gaps_from_scoreboard(baseline, source="scoreboard")

        goal, gap_ids = self.build_goal_from_gaps(limit=cfg.gap_limit)
        for gid in gap_ids:
            mark_gap_status(self.conn, gid, "in_progress")

        controller = RepoJanitorIterationController(
            repo_root=str(self.repo),
            db_conn=self.db,
            code_indexer=self.code_indexer,
            proposal_engine=self.proposal_engine,
            policy=self._default_policy(),
        )

        result = controller.run(goal=goal, budget=cfg.budget)

        # final scoreboard => gaps
        final = self.scoreboard.run(mode="all", fix=False)
        self.log_gaps_from_scoreboard(final, source="scoreboard")

        pr_url = None
        if cfg.open_pr and result.get("improved"):
            try:
                # controller already committed; just push
                self.pr_manager.push_branch(branch)

                body = (
                    "Repo Janitor unified self-improvement run\n\n"
                    f"### Goal\n{goal}\n\n"
                    f"### Baseline\nscore={result['baseline']['score']:.2f} gates={result['baseline']['gates']}\n\n"
                    f"### Best\nscore={result['best']['score']:.2f} gates={result['best']['gates']}\n\n"
                    f"Branch: `{branch}`\n"
                )

                pr_url = self.open_pr_from_branch(
                    base=base,
                    title=f"Repo Janitor: {branch_label}",
                    body=body,
                )

                try:
                    self.pr_manager.run_tests_and_update_pr(branch)
                except Exception:
                    pass
            except Exception as e:
                logger.exception(f"[self-improve] PR open failed: {e}")

        # mark gaps fixed if we improved and now passed
        if pr_url and final.passed():
            for gid in gap_ids:
                mark_gap_status(self.conn, gid, "fixed")
        else:
            # return them to queued so they remain visible
            for gid in gap_ids:
                mark_gap_status(self.conn, gid, "queued")

        try:
            self.pr_manager.restore_original_branch()
        except Exception:
            pass
          
          
        self._record_dream_summary_event(result=result, goal=goal)

        result["pr_url"] = pr_url
        result["branch"] = branch
        return result
      
      
      def _record_dream_summary_event(self, *, result: dict[str, Any], goal: str) -> None:
        """
        Persist a human-readable summary of the self-improve run into MemoryStore/events.
        This makes continuity visible and retrievable in normal conversation.
        """
        try:
            pr_url = result.get("pr_url") or ""
            branch = result.get("branch") or ""
            baseline = result.get("baseline") or {}
            best = result.get("best") or {}
            attempts = result.get("attempts") or []
            improved = bool(result.get("improved"))

            rollbacks = 0
            for a in attempts:
                if isinstance(a, dict) and a.get("improved") is False and a.get("error") is None:
                    # not perfect, but a good heuristic; you can refine later
                    pass

            # Pull the newest remaining gaps (still open)
            gaps = fetch_open_gaps(self.conn, limit=3)
            gap_lines = []
            for g in gaps:
                gap_lines.append(f"- ({g['classification']}) {g['requested_capability']} [prio={g['priority']}]")

            gap_block = "\n".join(gap_lines) if gap_lines else "- none"

            text = (
                "[dream_summary]\n"
                f"Goal:\n{goal}\n\n"
                f"Result: {'IMPROVED' if improved else 'NO CHANGE'}\n"
                f"Branch: {branch}\n"
                f"PR: {pr_url or '(none)'}\n\n"
                f"Baseline: score={baseline.get('score'):.2f} gates={baseline.get('gates')}\n"
                f"Best: score={best.get('score'):.2f} gates={best.get('gates')}\n"
                f"Attempts: {len(attempts)}\n\n"
                "Top remaining gaps:\n"
                f"{gap_block}\n"
            )

            if hasattr(self.store, "add_event"):
                self.store.add_event(content=text, importance=0.2, type_="dream_summary")
        except Exception as e:
            logger.debug(f"[self-improve] dream summary event skipped: {e}")