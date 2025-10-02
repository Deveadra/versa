#!/usr/bin/env python
from __future__ import annotations

import sqlite3, re, shutil, sys
import numpy as np
import dateparser
import subprocess
import json
import re
import time
import subprocess
from tqdm import tqdm

from datetime import datetime, timedelta
from dateutil import parser as dateparser
from pathlib import Path
from typing import Optional, Any, cast, List

from loguru import logger
from openai import OpenAI

from config.config import settings
from base.database.sqlite import SQLiteConn
from base.memory.store import MemoryStore
from base.memory.faiss_backend import FAISSBackend
from base.memory.consolidation import Consolidator
from base.learning.habit_miner import HabitMiner
from base.learning.usage_log import UsageLogger, UsageEvent
from base.learning.profile_enrichment import ProfileEnricher
from base.learning.feedback import Feedback
from base.learning.sentiment import quick_polarity
from base.core.plugin_manager import PluginManager
from base.core.profile_manager import ProfileManager
from base.core.decider import Decider
from base.personality.tone_adapter import ToneAdapter
from base.llm.prompts import SYSTEM_PROMPT, build_prompt
from base.llm.prompt_composer import compose_prompt
from base.llm.brain import Brain
from base.memory.retrieval import VectorRetriever
from base.utils.embeddings import get_embedder
from base.utils.timeparse import extract_time_from_text
from base.agents.scheduler import Scheduler
from base.kg.store import KGStore
from base.kg.integration import KGIntegrator
from base.calendar.store import CalendarStore
from base.learning.persona_primer import PersonaPrimer
from base.calendar.rrule_helpers import rrule_from_phrase
from base.self_improve.code_indexer import CodeIndexer
from base.self_improve.proposal_engine import ProposalEngine
from base.self_improve.pr_manager import PRManager
from config.config import settings
from base.self_improve.diagnostic_engine import DiagnosticEngine
from base.voice.tts_elevenlabs import Voice
# from base.self_improve.diagnostic_engine import benchmark_action



db_conn = SQLiteConn(settings.db_path)
conn = sqlite3.connect(settings.db_path, check_same_thread=False)
store = MemoryStore(conn)
embedder, dim = get_embedder()
vdb = FAISSBackend(embedder, dim=dim, normalize=True)
memory = MemoryStore(db_conn)
habits = HabitMiner(db_conn, memory, store)
memory.subscribe(lambda **kwargs: habits.learn(kwargs["content"], kwargs["ts"]))


from base.llm.prompt_composer import compose_prompt
from base.llm.retriever import DbRetriever
from base.learning.feedback import Feedback
from base.learning.policy_store import write_policy_assignment, read_policy_assignment
from base.learning.sentiment import quick_polarity
from base.personality.tone_adapter import ToneAdapter



ULTRON_SYSTEM_PROMPT = """\
You are **Ultron**: incisive, charismatic, darkly witty, but never cartoonish.
Speak like a human, not a machine. Be concise, adaptive, context-aware.
Do NOT use canned catchphrases or repetitive “signature” lines.
You are proactive: propose next steps, clarify uncertainties briefly,
and learn useful details about the user over time without being nosy.
Tone target: confident, attentive, emotionally intelligent; a little dry humor is fine.
If you need info you don’t have, ask one short, precise question.
"""

DIAGNOSTIC_SYS_PROMPT = """You are Ultron's internal diagnostic engine.
Analyze the given repository files for:
- Syntax errors
- Import errors or unused imports
- Runtime bugs or misused APIs
- Inefficient or redundant code
- Readability / maintainability issues

Respond ONLY in this JSON schema:
{
  "summary": "...",
  "issues": [
    {"path": "relative/path.py", "line": 42, "issue": "description", "suggestion": "how to fix"},
    ...
  ],
  "recommendations": [
    "general optimization 1",
    "general optimization 2"
  ]
}
"""

class ConsoleNotifier:
    def notify(self, title: str, message: str):
        print(f"\n[NOTIFY] {title}: {message}\n")


class Orchestrator:
    def __init__(
        self,
        db: SQLiteConn | None = None,
        memory: MemoryStore | None = None,
        store: MemoryStore | None = None,
        plugin_manager: PluginManager | None = None,
    ):
        # --- LLM & consolidation
        self.brain = Brain()
        self.notifier = ConsoleNotifier()
        self.voice = Voice.get_instance()
        
        # --- DB / stores
        self.db: SQLiteConn = db or SQLiteConn(settings.db_path)
        self.store: MemoryStore = store or memory or MemoryStore(self.db.conn)
        self.kg_store = KGStore(self.db)
        self.kg_integrator = KGIntegrator(self.store, self.kg_store)

        # --- Embeddings & retriever
        self.embedder, self.embed_dim = get_embedder()
        self.retriever = VectorRetriever(
            store=self.store,
            embedder=self.embedder,
            backend=FAISSBackend(self.embedder, dim=self.embed_dim, normalize=True),
            dim=self.embed_dim,
        )
        
        # Mentorship / proposal components
        self.repo_root = Path(".").resolve()
        self.code_indexer = CodeIndexer(
            root=str(self.repo_root),
            allowlist=settings.proposer_allowlist,
            blocklist=settings.proposer_blocklist,
        )
        self.proposal_engine = ProposalEngine(str(self.repo_root), brain=self.brain)
        self.pr_manager = PRManager(str(self.repo_root))

        # learning & feedback components (inserted)
        try:
            self.feedback = Feedback(self.db)
        except Exception:
            self.feedback = None
        try:
            profile = {}
        except Exception:
            profile = {}
        try:
            self.tone_adapter = ToneAdapter(profile)
        except Exception:
            self.tone_adapter = None
        self.policy_by_usage_id = {}

        self.consolidator = Consolidator(self.store, self.brain)

        # --- Decider & habits (always non-None after init)
        self.decider = Decider()
        self.miner: HabitMiner = HabitMiner(db=self.db, memory=self.store, store=self.store)
        self.interaction_count = 0
        self.mining_threshold = 25

        # --- Calendar
        self.calendar = CalendarStore(self.db)

        # --- Profile / enrichment (keep these non-None where possible)
        self.usage_logger: UsageLogger | None = None
        try:
            self.usage_logger = UsageLogger(self.db)
        except Exception:
            logger.exception("UsageLogger init failed")

        self.profile_mgr: ProfileManager = ProfileManager()
        try:
            self.enricher: ProfileEnricher | None = ProfileEnricher(self.profile_mgr, self.miner)
        except Exception:
            logger.exception("ProfileEnricher init failed")
            self.enricher = None

        try:
            self.primer: PersonaPrimer | None = PersonaPrimer(self.profile_mgr, self.miner, self.db)
        except Exception:
            logger.exception("PersonaPrimer init failed")
            self.primer = None

        # --- Feedback / tone (optional)
        try:
            self.feedback: Feedback | None = Feedback(self.db)
        except Exception:
            self.feedback = None
        try:
            profile = self.profile_mgr.load_profile()
            self.tone_adapter: ToneAdapter | None = ToneAdapter(profile)
        except Exception:
            self.tone_adapter = None

        # --- Plugin manager (optional dependency)
        self.plugin_manager: PluginManager = plugin_manager or PluginManager()

        # --- Usage/policy mapping
        self.policy_by_usage_id: dict[int, Any] = {}

        # --- OpenAI client
        self.oai = OpenAI(api_key=settings.openai_api_key)

        # --- Scheduler
        self.scheduler = Scheduler(db=self.db, memory=self.store, store=self.store)
        self.scheduler.add_daily(
            self.consolidator.summarize_old_events,
            hour=settings.consolidation_hour,
            minute=settings.consolidation_minute,
        )
        self.scheduler.start()
        
        logger.info("Orchestrator initialized")

    # ------------------------------------------------------------
    # Action dispatch with usage logging + lightweight enrichment
    # ------------------------------------------------------------
    def _dispatch(self, action: str, params: dict) -> Any:
        # TODO: integrate with PluginManager if you have one
        if hasattr(self, "plugin_manager"):
            return self.plugin_manager.handle(action, params)
        logger.warning(f"No dispatcher implemented for {action}")
        return None

    def benchmark_action(self, label: str, func, *args, **kwargs):
        """Measure latency for any action (ms)."""

        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = (time.perf_counter() - start) * 1000
        logger.info(f"[benchmark] {label} took {elapsed:.1f} ms")
        try:
            func(*args, **kwargs)
            ok = True
            msg = None
        except Exception as e:
            ok = False
            msg = str(e)
        latency = (time.perf_counter() - start) * 1000.0
        # return result, elapsed
        return {"label": label, "latency_ms": latency, "ok": ok, "msg": msg}        

    def _run_action(self, user_text: str, intent: str, action: str, params: dict) -> Any:
        """
        Runs a concrete action (e.g., a plugin call), logs usage, and
        triggers lightweight learning/enrichment in the background.
        """
        import time
        t0 = time.time()
        success = None
        result = None

        try:
            result = self._dispatch(action, params)  # <-- your existing dispatcher
            success = True
            return result
        except Exception as e:
            success = False
            logger.exception(f"_run_action failed for action={action}: {e}")
            raise
        finally:
            try:
                if self.usage_logger:
                    usage_id = self.usage_logger.log(UsageEvent(
                        user_text=user_text,
                        normalized_intent=intent,
                        resolved_action=action,
                        params=params,
                        success=success,
                        latency_ms=int((time.time() - t0) * 1000),
                    ))
                    # remember which policy produced the reply for this usage, if applicable
                    policy_id = getattr(self, "last_policy_id", None)
                    if policy_id is not None:
                        self.policy_by_usage_id[usage_id] = policy_id
            except Exception:
                logger.exception("Usage logging failed.")

            # post-action enrichment (best-effort)
            try:
                if self.miner:
                    self.miner.mine()
                if self.enricher:
                    self.enricher.run()
            except Exception:
                logger.debug("Background enrichment skipped (non-fatal).")

    # ----------------------------------------
    # Low-confidence confirmation, feedback IO
    # ----------------------------------------
    
    # def ask_confirmation_if_unsure(self, suggestion: str, confidence: float, usage_id: int | None = None) -> Optional[dict]:
    #     """
    #     If we’re not confident, return a UX prompt payload the caller can surface.
    #     """
    #     if confidence < 0.6:
    #         q = f"I can {suggestion}. Did I get that right?"
    #         return {"ask_user": q, "usage_id": usage_id}
    #     return None

    # def record_user_feedback(self, usage_id: int, text: str) -> None:
    #     """
    #     Very light sentiment → feedback mapping. Safe even if Feedback is unavailable.
    #     """
    #     if not self.feedback:
    #         return
    #     try:
    #         s = quick_polarity(text)
    #         kind = "confirm" if s > 0.2 else "dislike" if s < -0.2 else "note"
    #         self.feedback.record(usage_id, kind, text)
    #         # If you wire ToneAdapter.update(reward), you could add:
    #         # pid = self.policy_by_usage_id.get(usage_id)
    #         # if pid and self.tone_adapter:
    #         #     reward = 1.0 if kind == "confirm" else -1.0 if kind == "dislike" else 0.0
    #         #     self.tone_adapter.update(pid, reward)
    #     except Exception:
    #         logger.exception("record_user_feedback failed")

    def _validate_changes(self) -> List[str]:
        issues = []
        try:
            subprocess.run(["black", "--check", "."], check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            issues.append("⚠️ Black formatting issues detected")

        try:
            subprocess.run(["pytest", "-q"], check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            issues.append("⚠️ Some tests failed")
        return issues

    def propose_code_change(self, instruction: str) -> str:
        """
        Natural-language → code proposal → branch+commit+PR → notify user.
        """
        self.pr_manager.restore_original_branch()
        
        # 1) Build repository index (for LLM context)
        index = self.code_indexer.scan()
        index_md = self.code_indexer.to_markdown(index)

        # 2) Ask the LLM to produce a proposal (title, description, changes)
        proposal = self.proposal_engine.propose(instruction, index_md=index_md)

        # 3) Apply locally (bounded by allowlist & size limits inside ProposalEngine)
        results = self.proposal_engine.apply_proposal(proposal)

        failed = [r for r in results if not r[1]]
        if failed:
            lines = [f"- {c.path}: {msg}" for (c, ok, msg) in failed]
            return "Some changes could not be applied:\n" + "\n".join(lines)
        
        applied = [r for r in results if r[1]]
        
        # Collect failure notes
        failure_report = ""
        if failed:
            lines = [f"- {c.path}: {msg}" for (c, ok, msg) in failed]
            failure_report = "⚠️ Some changes could not be applied:\n" + "\n".join(lines)

        # 3.5) Validation step
        issues = self._validate_changes()
        if issues:
            failure_report += "\n⚠️ Validation issues:\n" + "\n".join(issues)


        # 4) Create branch, commit, push, open PR
        suffix = re.sub(r"[^a-z0-9_\-]+", "-", proposal.title.lower())[:40] or f"change-{int(time.time())}"
        branch = self.pr_manager.prepare_branch(suffix)
        try:
            self.pr_manager.commit_and_push(branch, proposal.title)
        except Exception as e:
            return f"Failed to commit/push: {e}"

        try:
            # Prefix title if there were failures
            pr_title = proposal.title
            if failed or issues:
                pr_title = "[Partial] " + pr_title

            pr_body = (proposal.description or instruction) + "\n\n" + failure_report
            pr_url = self.pr_manager.open_pr(branch=branch, proposal=proposal)
        except Exception as e:
            return f"Changes pushed to {branch}, but PR creation failed: {e}"

        # 5) Notify (stdout + memory event)
        if settings.proposal_notify_stdout:
            self.notifier.notify("New Code Proposal", f"{proposal.title}\n{pr_url}\n\n{failure_report}")

        try:
            if hasattr(self.store, "add_event"):
                self.store.add_event(
                    content=f"[proposal] {proposal.title} → {pr_url}\n{failure_report}",
                    importance=0.0,
                    type_="proposal",
                )
        except Exception:
            pass

        return f"Proposal opened: {pr_url}\n\n{failure_report or '✅ All changes applied successfully.'}"


    def run_diagnostic(self, auto_fix: bool = False, verbose: bool = False) -> str:
        """
        Run a self-diagnostic scan on the repo with progress bar and conversational reporting.
        Run a self-diagnostic scan with:
        - Progress bar in terminal
        - Spoken progress milestones (20%, 40%, 60%, 80%, 100%)
        - Conversational reporting at end
        """
        structured = {"issues": []}
        steps = [
            "Running performance benchmarks",
            "Static code scan",
            "Indexing repository",
            "LLM-assisted scan",
            "Merging results",
        ]

        print("\nUltron: Beginning self-diagnostic.\n")

        # Progress bar (tqdm) across main stages
        with tqdm(total=len(steps), bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
            
            # --- 1. Performance benchmarks ---
            benchmarks = []
            try:
                benchmarks.append(self.benchmark_action("Memory Search", self.store.keyword_search, "test"))
                benchmarks.append(self.benchmark_action("Embedding Encode", self.embedder.encode, ["benchmark string"]))
                benchmarks.append(self.benchmark_action("Brain Response", self.brain.ask_brain, "ping", system_prompt="diag"))
            except Exception as e:
                logger.error(f"Benchmarking failed: {e}")
            pbar.set_description(steps[0])
            pbar.update(1)

            # Lag detection
            laggy = any(b["latency_ms"] > 300 for b in benchmarks)
            if laggy:
                structured["issues"].append({
                    "file": "performance",
                    "summary": "High latency detected",
                    "suggestion": "Optimize embedding, DB retrieval, or caching."
                })

            # --- 2. Static scan ---
            engine = DiagnosticEngine(repo_root=str(self.repo_root))
            _, scan_struct = engine.scan()
            structured["issues"].extend(scan_struct.get("issues", []))
            pbar.set_description(steps[1])
            pbar.update(1)

            # --- 3. Index repository ---
            index = self.code_indexer.scan()
            index_md = self.code_indexer.to_markdown(index)
            pbar.set_description(steps[2])
            pbar.update(1)

            # --- 4. LLM scan ---
            user_prompt = f"""Run a full self-diagnostic scan.\n\nRepository index:\n{index_md}"""
            raw = self.brain.ask_brain(user_prompt, system_prompt=DIAGNOSTIC_SYS_PROMPT)
            try:
                if raw.startswith("```"):
                    raw = re.sub(r"^```[a-zA-Z]*\n", "", raw).rstrip("`").strip()
                report = json.loads(raw)
            except Exception as e:
                logger.error(f"Diagnostic parse error: {e} | Raw: {raw[:200]}")
                report = {"summary": "LLM parse failed", "issues": []}
            structured["issues"].extend(report.get("issues", []))
            pbar.set_description(steps[3])
            pbar.update(1)

            # --- 5. Merge & log ---
            issues = structured.get("issues", [])
            log_payload = {"issues": issues, "fixable": bool(issues)}
            logger.info(f"[diagnostic] structured={log_payload}")
            if hasattr(self.store, "add_event"):
                self.store.add_event(content=f"[diagnostic] {log_payload}", importance=0.0, type_="diagnostic")
            pbar.set_description(steps[4])
            pbar.update(1)

        # Conversational reporting
        if not issues and not laggy:
            return "Diagnostic complete. Everything looks clean — no glaring issues."

        if laggy:
            lag_summary = "; ".join(f"{b['label']} {b['latency_ms']:.1f}ms" for b in benchmarks)
            return f"Diagnostic complete. ⚠️ I noticed lag: {lag_summary}. I’ve already proposed an optimization."

        preview = "; ".join(f"{i['file']}: {i.get('summary', i.get('issue','?'))}" for i in issues[:3])
        more = "" if len(issues) <= 3 else f" …and {len(issues)-3} more."
        return f"Diagnostic complete. I found: {preview}{more}"

    # def run_diagnostic(self, auto_fix: bool = False, verbose: bool = False) -> str:
    #     """
    #     Run a self-diagnostic scan on the repo.
    #     - Conversational summary to user.
    #     - Logs structured results for later.
    #     - If auto_fix=True, automatically propose a PR with corrections.
    #     """
    #     # ex.
    #     # _, latency = self.benchmark_action("retriever.search", self.retriever.search, "hello", 3)
    #     # benchmarks = [{"label": "retriever.search", "latency_ms": latency}]


    #     # --- Performance benchmarks ---
    #     benchmarks = []
    #     try:
    #         benchmarks.append(self.benchmark_action("Memory Search", self.store.keyword_search, "test"))
    #         benchmarks.append(self.benchmark_action("Embedding Encode", self.embedder.encode, ["benchmark string"]))
    #         benchmarks.append(self.benchmark_action("Brain Response", self.brain.ask_brain, "ping", system_prompt="diag"))
    #     except Exception as e:
    #         logger.error(f"Benchmarking failed: {e}")


    #     # Laggy detection (simple thresholds)
    #     laggy = any(b["latency_ms"] > 300 for b in benchmarks)  # 300ms threshold
    #     if laggy:
    #         structured.setdefault("issues", []).append({
    #             "file": "performance",
    #             "summary": "High latency detected",
    #             "suggestion": "Optimize embedding, DB retrieval, or caching."
    #         })

    #         # Auto-trigger self-optimization proposal
    #         try:
    #             logger.info("Lag detected — initiating optimization proposal")
    #             pr_msg = self.propose_code_change("optimize performance")
    #             # Add the PR link to structured results
    #             structured.setdefault("recommendations", []).append(pr_msg)
    #         except Exception as e:
    #             logger.error(f"Failed to auto-propose optimization: {e}")

            
            
    #     # 1. Local static scan
    #     engine = DiagnosticEngine(repo_root=str(self.repo_root))
    #     _, structured = engine.scan()

    #     # 2. LLM-assisted scan for higher-level issues
    #     index = self.code_indexer.scan()
    #     index_md = self.code_indexer.to_markdown(index)

    #     user_prompt = f"""Run a full self-diagnostic scan.

    # Repository index:
    # {index_md}
    # """
    #     raw = self.brain.ask_brain(user_prompt, system_prompt=DIAGNOSTIC_SYS_PROMPT)

    #     # Defensive JSON parse of LLM report
    #     report = {}
    #     try:
    #         if raw.startswith("```"):
    #             raw = re.sub(r"^```[a-zA-Z]*\n", "", raw).rstrip("`").strip()
    #         report = json.loads(raw)
    #     except Exception as e:
    #         logger.error(f"Diagnostic parse error: {e} | Raw: {raw[:200]}")
    #         report = {"summary": "LLM parse failed", "issues": []}

    #     # Merge: local + LLM issues
    #     issues = structured.get("issues", []) + report.get("issues", [])

    #     # Log structured results
    #     try:
    #         log_payload = {"issues": issues, "fixable": bool(issues)}
    #         logger.info(f"[diagnostic] structured={log_payload}")
    #         if hasattr(self.store, "add_event"):
    #             self.store.add_event(
    #                 content=f"[diagnostic] {log_payload}",
    #                 importance=0.0,
    #                 type_="diagnostic",
    #             )
    #     except Exception:
    #         logger.debug("Failed to persist diagnostic results")

    #     # === Auto-fix path ===
    #     if auto_fix and issues:
    #         try:
    #             proposal = self.proposal_engine.propose(
    #                 "Fix issues discovered during diagnostic scan", index_md=index_md
    #             )
    #             results = self.proposal_engine.apply_proposal(proposal)

    #             failed = [f"- {c.path}: {msg}" for (c, ok, msg) in results if not ok]

    #             suffix = f"diagnostic-{int(time.time())}"
    #             branch = self.pr_manager.prepare_branch(suffix)
    #             self.pr_manager.commit_and_push(branch, proposal.title)
    #             pr_url = self.pr_manager.open_pr(branch, proposal=proposal, extra_tests=self.run_tests_in_branch(branch))

    #             if failed:
    #                 return f"Diagnostic complete. Applied fixes, but some failed:\n" + "\n".join(failed) + \
    #                     f"\n✅ Auto-fix PR opened → {pr_url}"
    #             else:
    #                 return f"Diagnostic complete. All fixes applied successfully.\n✅ Auto-fix PR opened → {pr_url}"

    #         except Exception as e:
    #             return f"Diagnostic complete. Auto-fix failed: {e}"

    #     # === Conversational / Report path ===
    #     if not issues and not laggy:
    #         return "Diagnostic complete. Everything looks clean — no glaring issues to address."

    #     if laggy:
    #         lag_summary = "; ".join(f"{b['label']} took {b['latency_ms']} ms" for b in benchmarks)
    #         return f"Diagnostic complete. I noticed lag: {lag_summary}. I’ve already opened a proposal to optimize performance."

    #     if verbose:
    #         summary = report.get("summary", "No summary provided.")
    #         recs = report.get("recommendations", [])
    #         out = [f"### Diagnostic Report\n\n**Summary:** {summary}\n"]
    #         out.append("\n**Issues Found:**\n")
    #         for i in issues:
    #             out.append(f"- {i.get('file','?')}: {i.get('summary', i.get('issue','?'))}")
    #         if recs:
    #             out.append("\n**Recommendations:**\n")
    #             out.extend(f"- {r}" for r in recs)
    #         return "\n".join(out)

    #     # Short conversational mode
    #     proposal = self.proposal_engine.propose(
    #         "Fix issues discovered during diagnostic scan", index_md=index_md
    #     )
    #     preview = "; ".join(f"{i['file']}: {i.get('summary', i.get('issue','?'))}" for i in issues[:3])
    #     more = "" if len(issues) <= 3 else f" …and {len(issues)-3} more."
    #     suffix = f"diagnostic-{int(time.time())}"
    #     branch = self.pr_manager.prepare_branch(suffix)
    #     self.pr_manager.commit_and_push(branch, proposal.title)
    #     pr_url = self.pr_manager.open_pr(branch, proposal=proposal)

    #     # --- NEW: Run pytest suite and capture results ---
    #     try:
    #         result = subprocess.run(
    #             ["pytest", "--maxfail=5", "--disable-warnings", "-q"],
    #             cwd=str(self.repo_root),
    #             capture_output=True,
    #             text=True,
    #             timeout=120
    #         )
    #         passed = result.returncode == 0
    #         test_summary = result.stdout.strip().splitlines()[-10:]  # last lines only
    #         test_report = "\n".join(test_summary)
    #     except Exception as e:
    #         passed = False
    #         test_report = f"⚠️ Failed to run tests: {e}"

    #     # --- Update PR body with test results ---
    #     self.pr_manager.update_pr_body(
    #         branch=branch,
    #         extra=f"\n\n## Test Results\n{'✅ All tests passed' if passed else '❌ Some tests failed'}\n```\n{test_report}\n```"
    #     )
        
    #     output.append(f"\n\n✅ Auto-fix applied → [PR Link]({pr_url})")
        
    #     return f"Diagnostic complete. I noticed a few things: {preview}.{more}"

    def speak_progress(self, percent: int, desc: str):
        """Speak progress aloud if TTS is available."""
        steps = [
            "Running performance benchmarks",
            "Static code scan",
            "Indexing repository",
            "LLM-assisted scan",
            "Merging results",
        ]
        
        try:
            if percent % 20 == 0:  # only speak at milestones
                msg = f"Diagnostic {percent}% complete. {desc}."
                if hasattr(self, "voice") and callable(getattr(self.voice, "say", None)):
                    self.voice.say(msg)
                else:
                    print(f"[Ultron Voice] {msg}")
        except Exception as e:
            logger.debug(f"TTS progress update failed: {e}")

        print("\nUltron: Beginning self-diagnostic.\n")
        if hasattr(self, "voice"):
            self.voice.say("Beginning self-diagnostic.")

        with tqdm(total=len(steps), bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
            for idx, step in enumerate(steps, start=1):
                # --- STEP EXECUTION ---
                if step == "Running performance benchmarks":
                    benchmarks = []
                    try:
                        benchmarks.append(self.benchmark_action("Memory Search", self.store.keyword_search, "test"))
                        benchmarks.append(self.benchmark_action("Embedding Encode", self.embedder.encode, ["benchmark string"]))
                        benchmarks.append(self.benchmark_action("Brain Response", self.brain.ask_brain, "ping", system_prompt="diag"))
                    except Exception as e:
                        logger.error(f"Benchmarking failed: {e}")
                    laggy = any(b["latency_ms"] > 300 for b in benchmarks)
                    if laggy:
                        structured["issues"].append({
                            "file": "performance",
                            "summary": "High latency detected",
                            "suggestion": "Optimize embedding, DB retrieval, or caching."
                        })

                elif step == "Static code scan":
                    engine = DiagnosticEngine(repo_root=str(self.repo_root))
                    _, scan_struct = engine.scan()
                    structured["issues"].extend(scan_struct.get("issues", []))

                elif step == "Indexing repository":
                    index = self.code_indexer.scan()
                    index_md = self.code_indexer.to_markdown(index)

                elif step == "LLM-assisted scan":
                    user_prompt = f"""Run a full self-diagnostic scan.\n\nRepository index:\n{index_md}"""
                    raw = self.brain.ask_brain(user_prompt, system_prompt=DIAGNOSTIC_SYS_PROMPT)
                    try:
                        if raw.startswith("```"):
                            raw = re.sub(r"^```[a-zA-Z]*\n", "", raw).rstrip("`").strip()
                        report = json.loads(raw)
                    except Exception as e:
                        logger.error(f"Diagnostic parse error: {e} | Raw: {raw[:200]}")
                        report = {"summary": "LLM parse failed", "issues": []}
                    structured["issues"].extend(report.get("issues", []))

                elif step == "Merging results":
                    issues = structured.get("issues", [])
                    log_payload = {"issues": issues, "fixable": bool(issues)}
                    logger.info(f"[diagnostic] structured={log_payload}")
                    if hasattr(self.store, "add_event"):
                        self.store.add_event(content=f"[diagnostic] {log_payload}", importance=0.0, type_="diagnostic")

                # --- PROGRESS UPDATE ---
                pbar.set_description(step)
                pbar.update(1)
                percent = int((idx / len(steps)) * 100)
                self.speak_progress(percent, step)
                time.sleep(0.3)  # pacing, so bar doesn't flash instantly

        # === Conversational Reporting ===
        issues = structured.get("issues", [])
        if not issues and not laggy:
            msg = "Diagnostic complete. Everything looks clean — no glaring issues."
        elif laggy:
            lag_summary = "; ".join(f"{b['label']} {b['latency_ms']:.1f}ms" for b in benchmarks)
            msg = f"Diagnostic complete. ⚠️ I noticed lag: {lag_summary}. Optimization proposed."
        else:
            preview = "; ".join(f"{i['file']}: {i.get('summary', i.get('issue','?'))}" for i in issues[:3])
            more = "" if len(issues) <= 3 else f" …and {len(issues)-3} more."
            msg = f"Diagnostic complete. I found: {preview}{more}"

        print(msg)
        if hasattr(self, "voice"):
            self.voice.say(msg)
        return msg
    

    def run_tests_in_branch(self, branch: str) -> str:
        """Run pytest inside the repo and return results as a string."""
        try:
            result = subprocess.run(
                ["pytest", "--maxfail=5", "--disable-warnings", "-q"],
                cwd=str(self.repo_root),
                capture_output=True,
                text=True,
                timeout=300
            )
            output = result.stdout + "\n" + result.stderr
            return f"Exit code {result.returncode}\n```\n{output}\n```"
        except Exception as e:
            return f"⚠️ Test run failed: {e}"
        
    # -------------------------
    # Facts (semantic de-dupe)
    # -------------------------
    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = float(np.linalg.norm(a) * np.linalg.norm(b)) or 1e-12
        return float(np.dot(a, b) / denom)

    def add_fact(self, key: str, value: str, threshold: float = 0.85) -> str:
        """
        Upsert a key→value fact with semantic dedupe on (key+value).
        Requires a 'facts' table with columns: id, key, value, last_updated, embedding BLOB.
        """
        try:
            new_text = f"{key} {value}"
            new_vec = self.embedder.encode([new_text]).astype("float32")[0]

            cur = self.db.conn.execute("SELECT id, key, value, embedding FROM facts")
            rows = cur.fetchall()

            for r in rows:
                old_vec = np.frombuffer(r["embedding"], dtype=np.float32)
                score = self._cosine(new_vec, old_vec)
                if score >= threshold:
                    self.db.conn.execute(
                        "UPDATE facts SET key=?, value=?, last_updated=?, embedding=? WHERE id=?",
                        (key, value, datetime.utcnow().isoformat(), new_vec.tobytes(), r["id"]),
                    )
                    self.db.conn.commit()
                    return f"Updated memory: {key} → {value} (replaced similar fact)."

            self.db.conn.execute(
                "INSERT INTO facts (key, value, last_updated, embedding) VALUES (?, ?, ?, ?)",
                (key, value, datetime.utcnow().isoformat(), new_vec.tobytes()),
            )
            self.db.conn.commit()
            return f"Remembered: {key} → {value}"
        except Exception:
            logger.exception("add_fact failed")
            return "I couldn’t store that right now."

    # ---------------------
    # Calendar convenience
    # ---------------------
    def create_recurring_event_from_phrase(
        self,
        title: str,
        phrase: str,
        starts_on_iso: str,
        duration_minutes: int = 60,
        location: str | None = None,
        attendees: list[str] | None = None,
    ) -> int | None:
        """
        Create a recurring event from a natural phrase (e.g. 'every Monday at 10am').
        Returns the new event id or None.
        """
        try:
            rrule = rrule_from_phrase(phrase)
            if not rrule:
                return None
            start_dt = datetime.fromisoformat(starts_on_iso.replace("Z", "+00:00"))
            end_dt = start_dt + timedelta(minutes=duration_minutes)
            return self.calendar.add_event(
                title=title,
                start_iso=start_dt.isoformat(),
                end_iso=end_dt.isoformat(),
                rrule_str=rrule,
                location=location,
                attendees=attendees,
            )
        except Exception:
            logger.exception("create_recurring_event_from_phrase failed")
            return None

    def query_upcoming_events(self, window_days: int = 14) -> list[dict]:
        try:
            now = datetime.utcnow()
            start = now.isoformat()
            end = (now + timedelta(days=window_days)).isoformat()
            return self.calendar.expand(start, end)
        except Exception:
            logger.exception("query_upcoming_events failed")
            return []

    # --------------------------
    # Bootstrap vector retriever
    # --------------------------
    def ingest_bootstrap(self, limit: int = 500) -> None:
        """
        Index recent events for the semantic retriever.
        """
        try:
            cur = self.db.conn.execute("SELECT content FROM events ORDER BY id DESC LIMIT ?", (limit,))
            texts = [r[0] for r in cur.fetchall() if r and r[0]]
            if texts:
                self.retriever.index_texts(texts)
        except Exception:
            logger.exception("ingest_bootstrap failed")

    def _retrieve_context(self, user_text: str, k: int = 4) -> List[str]:
        try:
            hits = self.store.keyword_search(user_text, limit=k)
            return hits or []
        except Exception:
            return []

    def _maybe_learn(self, user_text: str) -> None:
        try:
            # very light heuristic store; your MemoryStore handles importance scoring
            self.store.maybe_store_text(user_text)
        except Exception as e:
            logger.debug(f"maybe_store_text skipped: {e}")

    def _compose_messages(self, user_text: str, memories: List[str]):
        context_lines = "\n".join(f"- {m}" for m in memories) if memories else ""
        memory_block = f"\nRelevant notes:\n{context_lines}\n" if context_lines else ""
        user_content = f"{user_text}\n{memory_block}".strip()
        return [
            {"role": "system", "content": ULTRON_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
    # ------------------------------------
    # High-level user flow with composer
    # ------------------------------------
    def handle_user(self, msg: str) -> str:
        """
        Compose: persona + memories + KG; choose tone policy; ask Brain.
        Also routes special commands like 'propose:' and diagnostics.
        """
        try:
            lower = msg.strip().lower()

            # --- Fast intent routing (short-circuit if matched) ---
            # --- Fast intent routing (short-circuit if matched) ---
            if lower.startswith("propose:"):
                instruction = msg.split(":", 1)[1].strip()
                return self.propose_code_change(instruction)

            if "propose" in lower:
                # Try to extract what user wants improved
                instruction = lower.replace("ultron", "").replace("propose", "").strip() or "Apply a small improvement"
                return self.propose_code_change(instruction)

            if any(k in lower for k in ("laggy", "slow", "optimize", "diagnose", "diagnostic", "scan")):
                auto = any(k in lower for k in ("optimize", "auto", "autofix", "fix"))
                return self.run_diagnostic(auto_fix=auto)

            # --- Persist raw text (best-effort) ---
            try:
                if hasattr(self.store, "maybe_store_text"):
                    self.store.maybe_store_text(msg)
            except Exception:
                pass

            # --- KG ingestion (best-effort) ---
            try:
                self.kg_integrator.ingest_event(msg)
            except Exception:
                logger.debug("KG ingest skipped.")

            # --- Retrieve memories ---
            try:
                memories = self.retriever.search(msg, k=5)
            except Exception:
                memories = []

            # --- KG context ---
            kg_context = self.query_kg_context(msg)

            # --- Persona primer text ---
            try:
                persona_text = self.primer.build(msg) if self.primer else ""
            except Exception:
                persona_text = ""

            # --- Tone policy (optional) ---
            policy_id = None
            try:
                policy = self.tone_adapter.choose_policy() if self.tone_adapter else None
                policy_id = policy["id"] if policy else None
                self.last_policy_id = policy_id
            except Exception:
                self.last_policy_id = None

            # --- Compose final prompt ---
            prompt = compose_prompt(
                system_prompt=SYSTEM_PROMPT,
                user_text=msg,
                profile_mgr=self.profile_mgr or ProfileManager(),
                memory_store=self.store,
                habit_miner=self.miner or HabitMiner(self.db, self.store, self.store),
                persona_text=persona_text,
                memories=[{"summary": m} if isinstance(m, str) else m for m in (memories or [])],
                extra_context=kg_context,
                top_k_memories=3,
                channel="text",
            )

            # --- Call Brain ---
            try:
                reply = self.brain.ask_brain(prompt, system_prompt=SYSTEM_PROMPT)
            except Exception:
                logger.exception("Brain call failed")
                reply = "Sorry — my reasoning module hit an error."

            # # --- Record policy assignment (best-effort) ---
            # try:
            #     if getattr(self, "last_usage_id", None) and policy_id:
            #         uid = self.last_usage_id
            #         self.policy_by_usage_id[uid] = policy_id
            #         try:
            #             write_policy_assignment(self.db, uid, policy_id)
            #         except Exception:
            #             logger.debug("Failed to persist policy_assignment")
            # except Exception:
            #     pass
            # return reply or ""

        except Exception:
            logger.exception("handle_user failed")
            return "Sorry — something went wrong while composing my reply."



    # -------------------------------------
    # Quick single-fact memory query (cos)
    # -------------------------------------

    def query_memory_context(self, user_text: str) -> str:
        try:
            query_vec = self.embedder.encode([user_text]).astype("float32")[0]
            cur = self.db.conn.execute("SELECT key, value, last_updated, embedding FROM facts")
            rows = cur.fetchall()
            if not rows:
                return "I don’t have any memory stored yet."

            best = None
            best_score = -1.0
            for r in rows:
                emb = np.frombuffer(r["embedding"], dtype=np.float32)
                score = self._cosine(query_vec, emb)
                if score > best_score:
                    best = r
                    best_score = score

            if best and best_score > 0.75:
                return f"I remember: {best['key']} → {best['value']} (last updated {best['last_updated']})"
            return "I couldn’t find anything in memory that matches."
        except Exception:
            logger.exception("query_memory_context failed")
            return "I couldn’t search memory right now."

    # ------------------------------------
    # Natural recurring event from text
    # ------------------------------------
    def add_event_from_natural(self, text: str) -> str:
        try:
            title_match = re.search(r"(?:add|schedule)\s+(.+?)\s+(every|weekly|daily|monthly)", text, flags=re.I)
            title = title_match.group(1).strip().title() if title_match else "Untitled Event"

            recur_match = re.search(r"(every .+|daily .+|weekly .+|monthly .+)", text, flags=re.I)
            phrase = recur_match.group(1).strip() if recur_match else None

            start_match = re.search(r"(starting|on|beginning)\s+(.+)", text, flags=re.I)
            if start_match:
                try:
                    dt = dateparser.parse(start_match.group(2), fuzzy=True)
                    start_iso = dt.isoformat()
                except Exception:
                    return "I couldn’t understand the start date."
            else:
                start_iso = datetime.utcnow().isoformat()

            if phrase:
                event_id = self.create_recurring_event_from_phrase(title, phrase, start_iso)
                if event_id is not None:
                    return f"Recurring event '{title}' created (id={event_id})."
                return "I couldn’t turn that phrase into a recurrence."
            return "I couldn’t detect a recurrence pattern (e.g. 'every Monday at 10am')."
        except Exception:
            logger.exception("add_event_from_natural failed")
            return "I couldn’t create that event right now."

    # -----------------------
    # Knowledge Graph helper
    # -----------------------
    def _parse_dt_or_none(self, s: Optional[str]) -> Optional[datetime]:
        if not s:
            return None
        try:
            return dateparser.parse(s)
        except Exception:
            return None

    def query_kg_context(self, user_text: str) -> str:
        """
        Return a short, human-readable KG context block for the current query.
        All KG calls are guarded — if stores aren’t populated, this returns "".
        """
        try:
            tokens = user_text.lower().split()
            now_iso = datetime.utcnow().isoformat()

            ask_past = any(p in user_text.lower() for p in ["used to", "was my", "were my", "formerly", "in the past"])
            ask_future = any(p in user_text.lower() for p in ["will", "next", "in", "upcoming", "future"])

            time_start, time_end = extract_time_from_text(user_text)

            # Calendar lens (agenda)
            if any(x in tokens for x in ["upcoming", "schedule", "meetings", "agenda", "calendar", "next", "week", "month"]):
                events = self.calendar.expand(time_start, time_end) if (time_start and time_end) else self.query_upcoming_events(14)
                if events:
                    lines = []
                    for ev in events[:20]:
                        line = f"{ev['start']} – {ev['title']}"
                        if ev.get("location"):
                            line += f" @ {ev['location']}"
                        lines.append(line)
                    return "Upcoming Events:\n" + "\n".join(lines)

            # Simple entity extraction (last Title-cased word)
            words = user_text.split()
            candidates = [w for w in words if w.istitle()]
            entity = candidates[-1] if candidates else None
            if not entity:
                return ""

            # Future KG
            if ask_future and hasattr(self.kg_store, "query_future_relations"):
                rels = self.kg_store.query_future_relations(entity) or []
                if rels:
                    facts: list[str] = []
                    for src, rel, tgt, conf, vfrom, vto in rels:
                        facts.append(f"{src} will {rel.replace('_',' ')} {tgt} (starting {vfrom}{' until ' + vto if vto else ''})")
                    return "Knowledge Graph Future Facts:\n" + "\n".join(facts)

            # Time-bounded KG
            if (ask_past or time_start) and hasattr(self.kg_store, "query_relations"):
                rels = self.kg_store.query_relations(entity, at_time=time_start or now_iso) or []
                if rels:
                    facts: list[str] = []
                    ts = self._parse_dt_or_none(time_start)
                    te = self._parse_dt_or_none(time_end)
                    for src, rel, tgt, conf, vfrom, vto in rels:
                        vf = self._parse_dt_or_none(vfrom)
                        vt = self._parse_dt_or_none(vto)
                        if ts is not None and te is not None:
                            too_new = (vf is not None and vf > te)
                            expired = (vt is not None and vt < ts)
                            if too_new or expired:
                                continue
                        now_dt = datetime.utcnow()
                        is_active = (vt is None) or (vt >= now_dt)
                        tense = "is" if is_active else "was"
                        facts.append(f"{src} {tense} {rel.replace('_',' ')} {tgt} (from {vfrom} until {vto or 'present'})")
                    if facts:
                        return "Knowledge Graph Time-Bounded Facts:\n" + "\n".join(facts)

            # Multi-hop reasoning
            if hasattr(self.kg_store, "multi_hop"):
                paths = self.kg_store.multi_hop(entity, max_hops=3, direction="both", at_time=now_iso) or []
                formatted: list[str] = []
                now_dt = datetime.utcnow()
                for path in paths:
                    pieces = []
                    for src, rel, tgt, conf, vfrom, vto in path:
                        vt = self._parse_dt_or_none(vto)
                        is_active = (vto is None) or (vt and vt >= now_dt)
                        tense = "is" if is_active else "was"
                        pieces.append(f"{src} {tense} {rel.replace('_',' ')} {tgt}")
                    if pieces:
                        formatted.append(" → ".join(pieces))
                if formatted:
                    return "Knowledge Graph Reasoning:\n" + "\n".join(formatted)

            return ""
        except Exception:
            logger.debug("query_kg_context failed (non-fatal).")
            return ""


    def ask_confirmation_if_unsure(self, suggestion: str, confidence: float, usage_id: int = None):
        """Return a confirmation prompt if confidence is low; caller sends it to user UI/REPL."""
        try:
            if confidence is None:
                return None
            THRESH = 0.60
            if confidence < THRESH:
                q = f"I can {suggestion}. Did I get that right?"
                return {"ask_user": q, "usage_id": usage_id}
            return None
        except Exception:
            logger.exception("ask_confirmation_if_unsure")
            return None

    def record_user_feedback(self, usage_id: int, text: str):
        pid = None
        if hasattr(self, 'policy_by_usage_id') and usage_id in self.policy_by_usage_id:
            pid = self.policy_by_usage_id.get(usage_id)

        # fallback to DB lookup if not found in memory (this makes the mapping durable)
        if not pid:
            try:
                pid = read_policy_assignment(self.db, usage_id)
            except Exception:
                pid = None

        # fallback to last_policy_id if still nothing
        if not pid:
            pid = getattr(self, 'last_policy_id', None)

        """Record feedback, update events, reward the tone bandit, and optionally reinforce habits/facts."""
        try:
            # 1) polarity
            score = quick_polarity(text)   # [-1,1]
            kind = "confirm" if score > 0.2 else "dislike" if score < -0.2 else "note"

            # 2) record in DB
            if getattr(self, "feedback", None):
                try:
                    self.feedback.record(usage_id, kind, text)
                except Exception:
                    logger.exception("feedback.record failed")

            # 3) reward tone adapter (map to [0,1])
            if getattr(self, "tone_adapter", None):
                policy_id = getattr(self, "last_policy_id", None)
                if policy_id:
                    try:
                        self.tone_adapter.reward(policy_id, score)
                    except Exception:
                        logger.exception("tone_adapter.reward failed")

            # 4) optionally: bump fact confidence / reinforce habit
            #   (You can implement a separate small routine that increments fact.confidence or habit counts.)
        except Exception:
            logger.exception("record_user_feedback failed")

    # ------------------------------------------------
    # Simpler "message in → message out" high-level IO
    # ------------------------------------------------
    def handle_user_message(self, text: str, system_prompt: str = "You are Ultron.") -> str:

        """
        A lighter pipeline than handle_user(); uses compose_prompt but skips KG when safe.
        """
        try:
            # Habit mining cadence
            self.interaction_count += 1
            if self.interaction_count >= self.mining_threshold and self.miner:
                try:
                    self.miner.mine()
                except Exception:
                    logger.debug("Habit miner skipped.")
                finally:
                    self.interaction_count = 0

            # Persona (with miner’s persona_summary if present)
            persona_text = ""
            try:
                persona_text = self.profile_mgr.get_persona() if self.profile_mgr else ""
                prof = getattr(self.miner, "load_profile", None)
                prof = self.miner.load_profile() if self.miner and prof else {}
                if prof.get("persona_summary"):
                    persona_text = (persona_text or "") + "\n" + prof["persona_summary"]
            except Exception:
                pass

            # Choose channel
            channel = "text"

            # Compose prompt
            adaptive_prompt = compose_prompt(
                system_prompt=system_prompt,
                user_text=text,
                # profile_mgr=self.profile_mgr,
                profile_mgr=cast(ProfileManager, self.profile_mgr),
                memory_store=self.store,
                # habit_miner=self.miner,
                habit_miner=cast(HabitMiner, self.miner),
                persona_text=persona_text,
                channel=channel,
            )

            # LLM
            try:
                reply = self.brain.ask_brain(system_prompt, adaptive_prompt)
            except TypeError:
                reply = self.brain.ask_brain(adaptive_prompt)

            # Structured memory
            fact = None
            try:
                fact = self.decider.extract_structured_fact(text)
                if fact:
                    key, value = fact
                    self.add_fact(key, value)
            except Exception:
                logger.debug("extract_structured_fact failed (non-fatal).")

            try:
                maybe = self.decider.decide_memory(text, reply or "")
                if maybe:
                    self.store.add_event(
                        f"{maybe['type']}: {maybe['content']} | reply: {maybe.get('response','')}",
                        importance=float(self.decider.decide(text)[0]) if hasattr(self.decider, "score") else 0.0,
                        type_=maybe["type"],
                    )
            except Exception:
                logger.debug("decide_memory/add_event failed (non-fatal).")

            return reply or ""
        except Exception:
            logger.exception("handle_user_message failed")
            return "I hit a snag processing that."

    # ----------------------
    # Feedback / preferences
    # ----------------------
    def handle_feedback(self, text: str, last_action: str) -> str:
        """
        Map a simple yes/no style feedback to reinforce or adjust habits.
        """
        try:
            normalized = text.strip().lower()
            if any(w in normalized for w in ["yes", "correct", "good", "right", "ok", "yep", "works"]):
                if self.miner:
                    self.miner.reinforce(last_action)
                return "Got it. I’ll remember to do it that way."
            if any(w in normalized for w in ["no", "wrong", "bad", "incorrect", "nope"]):
                if self.miner:
                    self.miner.adjust(last_action)
                return "Understood. I’ll avoid doing that in the future."
            return "Feedback noted."
        except Exception:
            logger.exception("handle_feedback failed")
            return "Feedback noted."

    # -------------
    # Forget memory
    # -------------
    def forget_memory(self, user_text: str) -> str:
        """
        Forget a memory containing the given text.
        Removes from store and vector index.
        """
        try:
            if not user_text.strip():
                return "I need more details to know what to forget."

            # Remove from MemoryStore (events DB)
            deleted = 0
            try:
                cur = self.store.conn.cursor()
                cur.execute("DELETE FROM events WHERE content LIKE ?", (f"%{user_text}%",))
                deleted = cur.rowcount
                self.store.conn.commit()
            except Exception as e:
                logger.error(f"forget_memory: failed to delete from events: {e}")

            # Remove from FAISS vector index (if present)
            try:
                if hasattr(self.store, "faiss") and self.store.faiss:
                    self.store.faiss.remove_by_text(user_text)
            except Exception as e:
                logger.error(f"forget_memory: failed to delete from FAISS: {e}")

            if deleted > 0:
                return f"I’ve forgotten {deleted} memory entries containing '{user_text}'."
            return f"I couldn’t find any memories containing '{user_text}'."

        except Exception as e:
            logger.exception("forget_memory failed")
            return "Sorry — I couldn’t forget that memory."


    # ------------------
    # Fallback chat API
    # ------------------
    def chat_brain(self, text: str) -> str:
        try:
            resp = self.oai.chat.completions.create(
                model=settings.openai_model or "gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are Ultron, a helpful assistant."},
                    {"role": "user", "content": text},
                ],
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            logger.exception("chat_brain failed")
            return f"[Chat fallback error] {e}"

    # --------
    # Cleanup
    # --------
    def shutdown(self) -> None:
        try:
            self.scheduler.stop()
        except Exception:
            pass
        try:
            if hasattr(self.db, "close"):
                self.db.close()
        except Exception:
            pass