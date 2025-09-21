from __future__ import annotations
import numpy as np

import re
import sqlite3
import numpy as np
from datetime import datetime, timedelta
import re

from loguru import logger
from config.config import settings
from database.sqlite import SQLiteConn
import shutil
import sys
from datetime import datetime, timedelta
from dateutil import parser as dateparser
from loguru import logger

from base.core.profile_manager import ProfileManager
from base.learning.feedback import Feedback
from base.learning.habits import HabitMiner
from base.learning.profile_enrichment import ProfileEnricher
from base.learning.sentiment import quick_polarity
from base.learning.usage_log import UsageLogger, UsageEvent
from base.memory.store import MemoryStore
from ..memory.retrieval import Retriever
from ..embeddings.provider import Embeddings
from ..memory.retrieval import Retriever
from ..llm.brain import Brain
from ..llm.prompts import SYSTEM_PROMPT, build_prompt
from ..memory.consolidation import Consolidator
from .scheduler import Scheduler
from ..kg.store import KGStore
from ..kg.integration import KGIntegrator
from ..kg.relations import RELATION_QUERY_HINTS
from base.kg.store import KGStore
from ..calendar.rrule_helpers import rrule_from_phrase
from base.utils.timeparse import extract_time_from_text
from dateutil import parser as dateparser
from base.utils.embeddings import get_embedder
from base.memory.faiss_backend import FAISSBackend
from base.learning.habit_miner import HabitMiner
from base.utils.embeddings import get_embedder
from base.memory.decider import Decider

from .scheduler import Scheduler
from openai import OpenAI
from pathlib import Path


conn = sqlite3.connect(settings.db_path, check_same_thread=False)
store = MemoryStore(conn)
embedder, dim = get_embedder()
vdb = FAISSBackend(embedder, dim=dim, normalize=True)



def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    # robust cosine for 1-D vectors
    denom = float(np.linalg.norm(a) * np.linalg.norm(b)) or 1e-12
    return float(np.dot(a, b) / denom)


def _parse_dt_or_none(s: str | None):
    from dateutil import parser as dateparser
    if not s:
        return None
    try:
        return dateparser.parse(s)
    except Exception:
        return None

ROOT = Path(".")
TARGET = ROOT / "base" / "agents" / "orchestrator.py"
if not TARGET.exists():
    print("ERROR: orchestrator.py not found at:", TARGET)
    sys.exit(1)

bak = TARGET.with_suffix(".py.bak")
shutil.copy2(TARGET, bak)
print("Backup created:", bak)

txt = TARGET.read_text(encoding="utf-8", errors="ignore")

# 1) Add imports for Feedback, quick_polarity, ToneAdapter after existing composer/retriever import if present
if "from base.learning.feedback import Feedback" not in txt:
    anchor = "from base.llm.prompt_composer import compose_prompt"
    insert = ("from base.llm.prompt_composer import compose_prompt\n"
              "from base.llm.retriever import Retriever\n"
              "from base.learning.feedback import Feedback\n"
              "from base.learning.sentiment import quick_polarity\n"
              "from base.personality.tone_adapter import ToneAdapter\n")
    if anchor in txt:
        txt = txt.replace(anchor, insert, 1)
        print("Inserted imports after composer anchor.")
    else:
        # try alternate anchor
        anchor2 = "from base.llm.retriever import Retriever"
        if anchor2 in txt:
            txt = txt.replace(anchor2, insert, 1)
            print("Inserted imports after retriever anchor.")
        else:
            # fallback: put near top after module docstring or first import block
            m = re.search(r'(^\s*(?:import |from ).+\n)+', txt, flags=re.M)
            if m:
                pos = m.end()
                txt = txt[:pos] + "\n" + insert + txt[pos:]
                print("Inserted imports after top import block.")
            else:
                txt = insert + txt
                print("Prepended imports at file start (fallback).")

# 2) Enhance __init__ block: attempt to find existing block where PersonaPrimer is instantiated
if "self.primer = PersonaPrimer" in txt and "self.policy_by_usage_id" not in txt:
    # Insert the feedback/tone initialization after the line that sets self.primer
    txt = txt.replace(
        "self.primer = PersonaPrimer(self.profile_mgr, self.miner, self.db)",
        "self.primer = PersonaPrimer(self.profile_mgr, self.miner, self.db)\n"
        "            # optional feedback / tone components\n"
        "            try:\n"
        "                self.feedback = Feedback(self.db)\n"
        "            except Exception:\n"
        "                self.feedback = None\n"
        "            try:\n"
        "                profile = self.profile_mgr.load_profile() if self.profile_mgr else {}\n"
        "            except Exception:\n"
        "                profile = {}\n"
        "            try:\n"
        "                self.tone_adapter = ToneAdapter(profile)\n"
        "            except Exception:\n"
        "                self.tone_adapter = None\n"
        "            # mapping usage_id -> policy_id for async feedback attribution\n"
        "            self.policy_by_usage_id = {}\n"
    )
    print("Patched __init__ after PersonaPrimer instantiation.")
else:
    # fallback: try to insert near a previously added initial block 'self.primer = None' or after 'self.brain = Brain()'
    if "self.primer = None" in txt and "self.policy_by_usage_id" not in txt:
        txt = txt.replace("self.primer = None",
                          "self.primer = None\n            self.feedback = None\n            self.tone_adapter = None\n            self.policy_by_usage_id = {}\n")
        print("Patched fallback __init__ area (primer None).")
    elif "self.brain = Brain()" in txt and "self.policy_by_usage_id" not in txt:
        txt = txt.replace("self.brain = Brain()", "self.brain = Brain()\n" +
                          "        # learning & feedback components (inserted)\n" +
                          "        try:\n" +
                          "            self.feedback = Feedback(self.db)\n" +
                          "        except Exception:\n" +
                          "            self.feedback = None\n" +
                          "        try:\n" +
                          "            profile = {}\n" +
                          "        except Exception:\n" +
                          "            profile = {}\n" +
                          "        try:\n" +
                          "            self.tone_adapter = ToneAdapter(profile)\n" +
                          "        except Exception:\n" +
                          "            self.tone_adapter = None\n" +
                          "        self.policy_by_usage_id = {}\n")
        print("Inserted feedback init after self.brain = Brain() (fallback).")
    else:
        print("Warning: could not find suitable __init__ insertion point - you will need to add initialization manually.")
        # continue; we'll still try the other edits

# 3) Replace the call to LLM: detect a call to self.brain.complete(...prompt...) and replace it with policy selection + composer + mapping
if "reply = self.brain.complete(SYSTEM_PROMPT, prompt)" in txt and "self.policy_by_usage_id" in txt:
    replacement = (
        "# select tone policy (if available) and record it so we can attribute feedback later\n"
        "        try:\n"
        "            policy = self.tone_adapter.choose_policy() if getattr(self, \"tone_adapter\", None) else None\n"
        "            policy_id = policy[\"id\"] if policy else None\n"
        "            # stash last policy for immediate use (and mapping by usage_id later)\n"
        "            self.last_policy_id = policy_id\n"
        "        except Exception:\n"
        "            policy = None\n"
        "            policy_id = None\n"
        "            self.last_policy_id = None\n\n"
        "        # compose final prompt using the composer (persona + memories + extra_context)\n"
        "        prompt = compose_prompt(SYSTEM_PROMPT, user_text, persona_text=persona_text, memories=memories, extra_context=kg_context, top_k_memories=3)\n"
        "        # optionally: you may inject policy instructions into SYSTEM_PROMPT or extra_context based on policy here\n"
        "        reply = self.brain.complete(SYSTEM_PROMPT, prompt)\n\n"
        "        # if we logged a usage for this outgoing reply, attach mapping usage_id -> policy_id so feedback can credit the bandit\n"
        "        try:\n"
        "            if hasattr(self, \"last_usage_id\") and getattr(self, \"last_usage_id\", None):\n"
        "                uid = self.last_usage_id\n"
        "                if policy_id:\n"
        "                    try:\n"
        "                        self.policy_by_usage_id[uid] = policy_id\n"
        "                    except Exception:\n"
        "                        pass\n"
        "        except Exception:\n"
        "            pass\n"
    )
    txt = txt.replace("reply = self.brain.complete(SYSTEM_PROMPT, prompt)", replacement)
    print("Replaced LLM call with policy selection + composer + mapping.")
else:
    print("Warning: did not find exact 'reply = self.brain.complete(SYSTEM_PROMPT, prompt)' string; you may need to edit manually to integrate policy selection.")

# 4) Insert the two methods after __init__ end. Find insertion point: after def __init__ block (look for next 'def ' after it)
if "def ask_confirmation_if_unsure" not in txt:
    m = re.search(r"class\s+Orchestrator\b.*?def\s+__init__\s*\([^)]*\)\s*:\s*", txt, flags=re.S)
    if m:
        # locate end of __init__ by finding the next "\n\s*def\s" after m.end()
        rest = txt[m.end():]
        m2 = re.search(r"\n\s*def\s+", rest)
        if m2:
            insert_pos = m.end() + m2.start()
        else:
            # fallback: insert near the end of the class header region
            insert_pos = m.end()
        methods = ""
        
class Orchestrator:
    def __init__(self):
        # DBs / stores
        self.db = SQLiteConn(settings.db_path)
        self.store = MemoryStore(self.db)
        self.kg_store = KGStore(self.db)
        self.kg_integrator = KGIntegrator(self.store, self.kg_store)

        # Embeddings & retrieval
        self.embedder, self.embed_dim = get_embedder()
        self.retriever = Retriever(self.store, FAISSBackend(self.embedder, dim=self.embed_dim, normalize=True))

        # LLM & consolidation
        self.brain = Brain()
        self.consolidator = Consolidator(self.store, self.brain)

        # Learning
        self.miner = HabitMiner(self.db)
        self.interaction_count = 0
        self.mining_threshold = 25  # every 25 user interactions
        
        # Calendar
        self.calendar = CalendarStore(self.db)
            
        # Learning & Personality components
        try:
            self.usage_logger = UsageLogger(self.db)
            self.miner = HabitMiner(self.db)
            self.profile_mgr = ProfileManager()
            self.enricher = ProfileEnricher(self.profile_mgr, self.miner)
            self.primer = PersonaPrimer(self.profile_mgr, self.miner, self.db)
        except Exception:
            # non-fatal if learning modules are not available
            logger.exception("Learning components not initialized")
            self.usage_logger = None
            self.miner = None
            self.profile_mgr = None
            self.enricher = None
            self.primer = None

        # OpenAI v1 client for fallback chat
        self.oai = OpenAI(api_key=settings.openai_api_key)

        # Scheduler (use settings.consolidation_* names)
        self.scheduler = Scheduler(self.db)
        self.scheduler.add_daily(
            self.consolidator.summarize_old_events,
            hour=settings.consolidation_hour,
            minute=settings.consolidation_minute,
        )
        self.scheduler.start()


    def _run_action(self, user_text, intent, action, params):
        import time
        t0 = time.time()
        success = None
        try:
            result = self._dispatch(action, params)  # existing action runner
            success = True
            return result
        except Exception as e:
            success = False
            raise
        finally:
            usage_id = self.logger.log(UsageEvent(
                user_text=user_text,
                normalized_intent=intent,
                resolved_action=action,
                params=params,
                success=success,
                latency_ms=int((time.time()-t0)*1000),
            ))
            # lightweight background maintenance
            try:
                self.miner.update_from_usage()
                self.enricher.run()
            except Exception:
                pass


    def ask_confirmation_if_unsure(self, suggestion: str, confidence: float, usage_id: int):
        if confidence < 0.6:
            q = f"I can {suggestion}. Did I get that right?"
            # send to REPL/UX
            return {"ask_user": q, "usage_id": usage_id}

    def record_user_feedback(self, usage_id: int, text: str):
        s = quick_polarity(text)
        kind = "confirm" if s > 0.2 else "dislike" if s < -0.2 else "note"
        self.feedback.record(usage_id, kind, text)
        # update personality bandit via ToneAdapter outside (when crafting replies)
        
            # ---------- Facts (semantic dedupe) ----------
    def add_fact(self, key: str, value: str, threshold: float = 0.85) -> str:
        new_text = f"{key} {value}"
        new_vec = self.embedder.encode([new_text]).astype("float32")[0]

        cur = self.db.conn.execute("SELECT id, key, value, embedding FROM facts")
        rows = cur.fetchall()

        for r in rows:
            old_vec = np.frombuffer(r["embedding"], dtype=np.float32)
            score = _cosine(new_vec, old_vec)
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

    # ---------- Calendar helpers ----------
    def create_recurring_event_from_phrase(
        self,
        title: str,
        phrase: str,
        starts_on_iso: str,
        duration_minutes: int = 60,
        location: str | None = None,
        attendees: list[str] | None = None,
    ) -> int | None:
        rrule = rrule_from_phrase(phrase)
        if not rrule:
            print("[Calendar] Could not parse recurrence phrase.")
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

    # ----- CALENDAR: query window -----
    def query_upcoming_events(self, window_days: int = 14) -> list[dict]:
        now = datetime.utcnow()
        start = now.isoformat()
        end = (now + timedelta(days=window_days)).isoformat()
        return self.calendar.expand(start, end)

    # ---------- Bootstrap ----------
    def ingest_bootstrap(self):
        cur = self.db.conn.execute("SELECT content FROM events ORDER BY id DESC LIMIT 500")
        texts = [r[0] for r in cur.fetchall()]
        if texts:
            self.retriever.index(texts)

    # ---------- Main user handling ----------
    def handle_user(self, user_text: str) -> str:
        self.store.maybe_store_text(user_text)
        self.kg_integrator.ingest_event(user_text)

        memories = self.retriever.search(user_text, k=5)
        kg_context = self.query_kg_context(user_text)

        prompt = build_prompt(memories, user_text, extra_context=kg_context)
        # Insert persona primer into prompt extra context (if available)
        try:
            persona_text = self.primer.build(user_text) if self.primer else ''
            if persona_text:
                prompt = build_prompt(memories, user_text, extra_context=kg_context + '\nPersona: ' + persona_text)
        except Exception:
            prompt = build_prompt(memories, user_text, extra_context=kg_context)
        reply = self.brain.complete(SYSTEM_PROMPT, prompt)
        return reply

    # ---------- Memory recall (semantic) ----------
    def query_memory_context(self, user_text: str) -> str:
        query_vec = self.embedder.encode([user_text]).astype("float32")[0]

        cur = self.db.conn.execute("SELECT key, value, last_updated, embedding FROM facts")
        rows = cur.fetchall()
        if not rows:
            return "I don’t have any memory stored yet."

        best = None
        best_score = -1.0
        for r in rows:
            emb = np.frombuffer(r["embedding"], dtype=np.float32)
            score = _cosine(query_vec, emb)
            if score > best_score:
                best = r
                best_score = score

        if best and best_score > 0.75:
            return f"I remember: {best['key']} → {best['value']} (last updated {best['last_updated']})"
        return "I couldn’t find anything in memory that matches."

    # ---------- Natural calendar parsing ----------
    def add_event_from_natural(self, text: str) -> str:
        title_match = re.search(r"(?:add|schedule) (.+?) (every|weekly|daily|monthly)", text.lower())
        title = title_match.group(1).title() if title_match else "Untitled Event"

        recur_match = re.search(r"(every .+|daily .+|weekly .+|monthly .+)", text.lower())
        phrase = recur_match.group(1) if recur_match else None

        start_match = re.search(r"(starting|on|beginning) (.+)", text.lower())
        if start_match:
            from dateutil import parser as dateparser
            try:
                dt = dateparser.parse(start_match.group(2), fuzzy=True)
                start_iso = dt.isoformat()
            except Exception:
                return "I couldn’t understand the start date."
        else:
            start_iso = datetime.utcnow().isoformat()

        if phrase:
            event_id = self.create_recurring_event_from_phrase(title, phrase, start_iso)
            return f"Recurring event '{title}' created (id={event_id})."
        return "I couldn’t detect the recurrence pattern (e.g. 'every Monday at 10am')."

    # ---------- KG context ----------
    def query_kg_context(self, user_text: str) -> str:
        tokens = user_text.lower().split()
        now_iso = datetime.utcnow().isoformat()

        ask_past = any(p in user_text.lower() for p in ["used to", "was my", "were my", "used be", "formerly", "in the past"])
        ask_future = any(p in user_text.lower() for p in ["will", "next", "in", "upcoming", "future"])

        # explicit time reference (single place to call)
        time_start, time_end = extract_time_from_text(user_text)

        # Event queries
        if any(x in tokens for x in ["upcoming", "schedule", "meetings", "agenda", "calendar", "next", "week", "month"]):
            if time_start and time_end:
                events = self.calendar.expand(time_start, time_end)
            else:
                events = self.query_upcoming_events(window_days=14)

            if events:
                lines = []
                for ev in events[:20]:
                    lines.append(f"{ev['start']} – {ev['title']}" + (f" @ {ev['location']}" if ev.get('location') else ""))
                return "Upcoming Events:\n" + "\n".join(lines)

        # Future KG facts
        if ask_future:
            words = user_text.split()
            candidates = [w for w in words if w.istitle()]
            if candidates:
                entity = candidates[-1]
                rels = self.kg_store.query_future_relations(entity)
                if rels:
                    facts: list[str] = []
                    for src, rel, tgt, conf, vfrom, vto in rels:  # 6-tuple
                        facts.append(
                            f"{src} will {rel.replace('_',' ')} {tgt} "
                            f"(starting {vfrom}{' until ' + vto if vto else ''})"
                        )
                    return "Knowledge Graph Future Facts:\n" + "\n".join(facts)

        # General KG reasoning
        if any(t in tokens for t in ["who", "relation", "related", "about", "husband", "wife", "parent", "child", "boss", "mom", "dad", "work", "job", "sleep"]):
            words = user_text.split()
            candidates = [w for w in words if w.istitle()]
            if candidates:
                entity = candidates[-1]

                if ask_past or time_start:
                    rels = self.kg_store.query_relations(entity, at_time=time_start or now_iso)
                    if rels:
                        facts: list[str] = []
                        ts = _parse_dt_or_none(time_start)
                        te = _parse_dt_or_none(time_end)
                        for src, rel, tgt, conf, vfrom, vto in rels:
                            vf = _parse_dt_or_none(vfrom)
                            vt = _parse_dt_or_none(vto)
                            # only include if it overlaps requested window
                            if ts is not None and te is not None:
                                too_new = (vf is not None and vf > te)
                                expired = (vt is not None and vt < ts)
                                if too_new or expired:
                                    continue
                            now_dt = datetime.utcnow()
                            tense = "is" if (vt is None or vt >= now_dt) else "was"
                            facts.append(
                                f"{src} {tense} {rel.replace('_',' ')} {tgt} "
                                f"(from {vfrom} until {vto or 'present'})"
                            )
                        if facts:
                            return "Knowledge Graph Time-Bounded Facts:\n" + "\n".join(facts)
                        return "No facts found for that time frame."

                paths = self.kg_store.multi_hop(entity, max_hops=3, direction="both", at_time=now_iso)
                formatted: list[str] = []
                now_dt = datetime.utcnow()
                for path in paths:
                    pieces = []
                    for src, rel, tgt, conf, vfrom, vto in path:
                        vt = _parse_dt_or_none(vto)
                        is_active = (vto is None) or (vt is not None and vt >= now_dt)
                        tense = "is" if is_active else "was"
                        pieces.append(f"{src} {tense} {rel.replace('_',' ')} {tgt}")
                    formatted.append(" → ".join(pieces))
                if formatted:
                    return "Knowledge Graph Reasoning:\n" + "\n".join(formatted)

        return ""
    
    # ---------- User interaction & learning ----------
    def handle_user_message(self, text: str) -> str:
        """
        Main entrypoint for processing user input.
        """
        reply = self.llm.generate(text)
        maybe = decider.decide_memory(text, reply)
        if maybe:
            self.store.save_memory(maybe)
            return reply
        
        # ✅ Count interaction
        self.interaction_count += 1
        if self.interaction_count >= self.mining_threshold:
            try:
                logger.info("Triggering HabitMiner (interaction threshold reached)")
                self.miner.mine()
            except Exception as e:
                logger.error(f"HabitMiner mining failed: {e}")
            finally:
                self.interaction_count = 0  # reset

        return reply

    # ---------- Misc ----------
    def forget_memory(self, user_text: str) -> str:
        words = user_text.split()
        target = next((w for w in words if w.istitle()), None)
        if not target:
            return "What should I forget?"
        self.db.conn.execute("DELETE FROM facts WHERE key LIKE ?", (f"%{target}%",))
        self.db.conn.commit()
        return f"I’ve forgotten what I knew about {target}."

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
            return f"[ChatGPT Fallback Error] {e}"

    def shutdown(self) -> None:
        self.scheduler.stop()
        self.db.close()
