#!/usr/bin/env python
from __future__ import annotations

import sqlite3, re, shutil, sys
import numpy as np
import dateparser
from datetime import datetime, timedelta
from dateutil import parser as dateparser
from pathlib import Path
from typing import Optional, Any, cast

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
from base.memory.retrieval import Retriever
from base.utils.embeddings import get_embedder
from base.utils.timeparse import extract_time_from_text
from base.agents.scheduler import Scheduler
from base.kg.store import KGStore
from base.kg.integration import KGIntegrator
from base.calendar.store import CalendarStore
from base.learning.persona_primer import PersonaPrimer
from base.calendar.rrule_helpers import rrule_from_phrase


db_conn = SQLiteConn(settings.db_path)
conn = sqlite3.connect(settings.db_path, check_same_thread=False)
store = MemoryStore(conn)
embedder, dim = get_embedder()
vdb = FAISSBackend(embedder, dim=dim, normalize=True)
memory = MemoryStore(db_conn)
habits = HabitMiner(db_conn, memory, store)
memory.subscribe(lambda **kwargs: habits.learn(kwargs["content"], kwargs["ts"]))


from base.llm.prompt_composer import compose_prompt
from base.llm.retriever import Retriever
from base.learning.feedback import Feedback
from base.learning.policy_store import write_policy_assignment, read_policy_assignment
from base.learning.sentiment import quick_polarity
from base.personality.tone_adapter import ToneAdapter

class Orchestrator:
    def __init__(
        self,
        db: SQLiteConn | None = None,
        memory: MemoryStore | None = None,
        store: MemoryStore | None = None,
        plugin_manager: PluginManager | None = None,
    ):
        # --- DB / stores
        self.db: SQLiteConn = db or SQLiteConn(settings.db_path)
        self.store: MemoryStore = store or memory or MemoryStore(self.db.conn)
        self.kg_store = KGStore(self.db)
        self.kg_integrator = KGIntegrator(self.store, self.kg_store)

        # --- Embeddings & retriever
        self.embedder, self.embed_dim = get_embedder()
        self.retriever = Retriever(
            self.store,
            FAISSBackend(self.embedder, dim=self.embed_dim, normalize=True),
        )

        # --- LLM & consolidation
        self.brain = Brain()
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

        
        
    # ------------------------------------------------------------
    # Action dispatch with usage logging + lightweight enrichment
    # ------------------------------------------------------------
    def _dispatch(self, action: str, params: dict) -> Any:
        # TODO: integrate with PluginManager if you have one
        if hasattr(self, "plugin_manager"):
            return self.plugin_manager.handle(action, params)
        logger.warning(f"No dispatcher implemented for {action}")
        return None


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
    def ask_confirmation_if_unsure(self, suggestion: str, confidence: float, usage_id: int | None = None) -> Optional[dict]:
        """
        If we’re not confident, return a UX prompt payload the caller can surface.
        """
        if confidence < 0.6:
            q = f"I can {suggestion}. Did I get that right?"
            return {"ask_user": q, "usage_id": usage_id}
        return None

    def record_user_feedback(self, usage_id: int, text: str) -> None:
        """
        Very light sentiment → feedback mapping. Safe even if Feedback is unavailable.
        """
        if not self.feedback:
            return
        try:
            s = quick_polarity(text)
            kind = "confirm" if s > 0.2 else "dislike" if s < -0.2 else "note"
            self.feedback.record(usage_id, kind, text)
            # If you wire ToneAdapter.update(reward), you could add:
            # pid = self.policy_by_usage_id.get(usage_id)
            # if pid and self.tone_adapter:
            #     reward = 1.0 if kind == "confirm" else -1.0 if kind == "dislike" else 0.0
            #     self.tone_adapter.update(pid, reward)
        except Exception:
            logger.exception("record_user_feedback failed")

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
                self.retriever.index(texts)
        except Exception:
            logger.exception("ingest_bootstrap failed")

    # ------------------------------------
    # High-level user flow with composer
    # ------------------------------------
    def handle_user(self, user_text: str) -> str:
        """
        Compose: persona + memories + KG; choose tone policy; ask Brain.
        """
        try:
            # Persist raw text (if your store does this)
            try:
                if hasattr(self.store, "maybe_store_text"):
                    self.store.maybe_store_text(user_text)
            except Exception:
                pass

            # KG ingestion of event
            try:
                self.kg_integrator.ingest_event(user_text)
            except Exception:
                logger.debug("KG ingest skipped.")

            # Retrieve memories
            try:
                memories = self.retriever.search(user_text, k=5)  # returns texts or dicts depending on your Retriever
            except Exception:
                memories = []

            # KG context
            kg_context = self.query_kg_context(user_text)

            # Persona
            try:
                persona_text = self.primer.build(user_text) if self.primer else ""
            except Exception:
                persona_text = ""

            # Tone policy (optional bandit)
            try:
                policy = self.tone_adapter.choose_policy() if self.tone_adapter else None
                self.last_policy_id = policy["id"] if policy else None
            except Exception:
                self.last_policy_id = None

            # Compose final prompt
            prompt = compose_prompt(
                system_prompt=SYSTEM_PROMPT,
                user_text=user_text,
                # profile_mgr=self.profile_mgr,
                profile_mgr=cast(ProfileManager, self.profile_mgr),
                memory_store=self.store,
                # habit_miner=self.miner,
                habit_miner=cast(HabitMiner, self.miner),
                persona_text=persona_text,
                memories=[{"summary": m} if isinstance(m, str) else m for m in (memories or [])],
                extra_context=kg_context,
                top_k_memories=3,
                channel="text",
            )


        # Build prompt with KG facts included
        prompt = build_prompt(memories, user_text, extra_context=kg_context)
        # select tone policy (if available) and record it so we can attribute feedback later
        try:
            policy = self.tone_adapter.choose_policy() if getattr(self, "tone_adapter", None) else None
            policy_id = policy["id"] if policy else None
            # stash last policy for immediate use (and mapping by usage_id later)
            self.last_policy_id = policy_id
        except Exception:
            policy = None
            policy_id = None
            self.last_policy_id = None

        # compose final prompt using the composer (persona + memories + extra_context)
        prompt = compose_prompt(SYSTEM_PROMPT, user_text, persona_text=persona_text, memories=memories, extra_context=kg_context, top_k_memories=3)
        # optionally: you may inject policy instructions into SYSTEM_PROMPT or extra_context based on policy here
        reply = self.brain.complete(SYSTEM_PROMPT, prompt)

        # if we logged a usage for this outgoing reply, attach mapping usage_id -> policy_id so feedback can credit the bandit
        # if we logged a usage for this outgoing reply, attach mapping usage_id -> policy_id so feedback can credit the bandit
        try:
            if hasattr(self, "last_usage_id") and getattr(self, "last_usage_id", None):
                uid = self.last_usage_id
                if policy_id:
                    try:
                        # keep in-memory mapping (fast)
                        self.policy_by_usage_id[uid] = policy_id
                    except Exception:
                        pass
                    # persist durable mapping to DB (robust across restarts/workers)
                    try:
                        write_policy_assignment(self.db, uid, policy_id)
                    except Exception:
                        try:
                            logger.exception("Failed to persist policy_assignment")
                        except Exception:
                            pass
        except Exception:
            pass


        return reply
    

            # LLM call (Brain supports either complete(system, prompt) or complete(prompt))
            try:
                reply = self.brain.ask_brain(prompt)
            except TypeError:
                reply = self.brain.ask_brain(prompt)

            return reply or ""
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
            

    def forget_memory(self, user_text: str) -> str:

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
        try:
            words = user_text.split()
            target = next((w for w in words if w.istitle()), None)
            if not target:
                return "What should I forget?"
            self.db.conn.execute("DELETE FROM facts WHERE key LIKE ?", (f"%{target}%",))
            self.db.conn.commit()
            return f"I’ve forgotten what I knew about {target}."
        except Exception:
            logger.exception("forget_memory failed")
            return "I couldn’t forget that right now."

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