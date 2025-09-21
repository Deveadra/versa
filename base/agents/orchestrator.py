from __future__ import annotations

import re
import sqlite3
import numpy as np
from datetime import datetime, timedelta

from loguru import logger
from config.config import settings
from database.sqlite import SQLiteConn
from base.memory.store import MemoryStore
from ..memory.retrieval import Retriever
from ..embeddings.provider import Embeddings
from ..llm.brain import Brain
from ..memory.faiss_backend import FAISSBackend
from ..llm.prompts import SYSTEM_PROMPT, build_prompt
from ..memory.consolidation import Consolidator
from .scheduler import Scheduler
from ..kg.store import KGStore
from ..kg.integration import KGIntegrator
from ..kg.relations import RELATION_QUERY_HINTS
from datetime import datetime, timedelta
from base.calendar.store import CalendarStore
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
