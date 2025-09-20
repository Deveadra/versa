
from __future__ import annotations
import openai

from loguru import logger
from assistant.config.config import settings
from database.sqlite import SQLiteConn
from base.memory.store import MemoryStore
from base.memory.store import maybe_store_text
from ..memory.retrieval import Retriever
from ..embeddings.provider import Embeddings
from ..llm.brain import Brain
from ..memory.faiss_backend import FAISSBackend
from ..llm.prompts import SYSTEM_PROMPT, build_prompt
from ..memory.consolidation import Consolidator
from .scheduler import UltronScheduler
from ..kg.store import KGStore
from ..kg.integration import KGIntegrator
from ..kg.relations import RELATION_QUERY_HINTS
from datetime import datetime, timedelta
from ..calendar.store import CalendarStore
from ..calendar.rrule_helpers import rrule_from_phrase
from dateutil import parser as dateparser
from ..utils.embeddings import get_embedding
import numpy as np
from ..utils.embeddings import get_embedding, cosine_similarity
from ..utils.embeddings import get_embedding, cosine_similarity

from base.llm.prompt_composer import compose_prompt
from base.llm.retriever import Retriever
from base.learning.feedback import Feedback
from base.learning.policy_store import write_policy_assignment, read_policy_assignment
from base.learning.sentiment import quick_polarity
from base.personality.tone_adapter import ToneAdapter

class Orchestrator:
    def __init__(self):
        self.db = SQLiteConn(settings.db_path)
        self.store = MemoryStore(self.db)
        self.kg_store = KGStore(self.db)
        self.kg_integrator = KGIntegrator(self.store, self.kg_store)
        self.embedder = Embeddings(settings.embeddings_model) if settings.embeddings_provider else None
        backend = FAISSBackend(self.embedder, dim=384) if self.embedder else None
        self.retriever = Retriever(self.store, backend)
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
        self.calendar = CalendarStore(self.db)

        # Scheduler (configurable via .env)
        self.scheduler = UltronScheduler()
        self.scheduler.add_daily(
            self.consolidator.summarize_old_events,
            hour=settings.cron_hour,
            minute=settings.cron_minute,
        )
        self.scheduler.start()


    def add_fact(self, key: str, value: str, threshold: float = 0.85) -> str:
        """
        Add a fact to memory with semantic deduplication.
        If a semantically similar fact exists, update it instead of duplicating.
        """
        new_text = f"{key} {value}"
        new_emb = get_embedding(new_text)

        # search for similar facts
        cur = self.db.conn.execute("SELECT id, key, value, embedding FROM facts")
        rows = cur.fetchall()

        for r in rows:
            old_emb = np.frombuffer(r["embedding"], dtype=np.float32)
            score = cosine_similarity(new_emb, old_emb)

            if score >= threshold:
                # Update existing fact instead of duplicating
                self.db.conn.execute(
                    "UPDATE facts SET key=?, value=?, last_updated=?, embedding=? WHERE id=?",
                    (key, value, datetime.utcnow().isoformat(), new_emb.tobytes(), r["id"])
                )
                self.db.conn.commit()
                return f"Updated memory: {key} → {value} (replaced similar fact)."

        # No duplicate found, insert new fact
        self.db.conn.execute(
            "INSERT INTO facts (key, value, last_updated, embedding) VALUES (?, ?, ?, ?)",
            (key, value, datetime.utcnow().isoformat(), new_emb.tobytes())
        )
        self.db.conn.commit()
        return f"Remembered: {key} → {value}"
        
        
# ----- CALENDAR: create recurring event -----
    def create_recurring_event_from_phrase(
        self, title: str, phrase: str, starts_on_iso: str, duration_minutes: int = 60,
        location: str | None = None, attendees: list[str] | None = None
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
    
    
    def ingest_bootstrap(self):
        cur = self.db.conn.execute("SELECT content FROM events ORDER BY id DESC LIMIT 500")
        texts = [r[0] for r in cur.fetchall()]
        if texts:
            self.retriever.index(texts)
    
    def handle_user(self, user_text: str) -> str:
        self.store.maybe_store_text(user_text)
        self.kg_integrator.ingest_event(user_text)


        # Gather memories + KG reasoning
        memories = self.retriever.search(user_text, k=5)
        kg_context = self.query_kg_context(user_text)


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
    
    def query_memory_context(self, user_text: str) -> str:
        """
        Semantic memory recall using embeddings.
        Handles natural memory queries like:
        - 'Do you remember what time I went to sleep last Friday?'
        - 'What did I tell you about Alice’s favorite color?'
        """
        
        # encode the query
        query_vec = get_embedding(user_text)
    
        # For MVP, just do a simple search. Later: embeddings.
        cur = self.db.conn.execute("SELECT key, value, last_updated, embedding FROM facts")
        # cur = self.db.conn.execute(
        #     "SELECT key, value, last_updated FROM facts ORDER BY last_updated DESC LIMIT 20"
        # )
        rows = cur.fetchall()
        if not rows:
            return "I don’t have any memory stored yet."

        best = None
        best_score = -1.0
    
        # naive keyword match
        matches = []
        for r in rows:
            emb = np.frombuffer(r["embedding"], dtype=np.float32)
            score = cosine_similarity(query_vec, emb)
            if score > best_score:
                best = r
                best_score = score

        if best and best_score > 0.75:  # threshold
            return f"I remember: {best['key']} → {best['value']} (last updated {best['last_updated']})"
        else:
            return "I couldn’t find anything in memory that matches."
        
        # for r in rows:
        #     if r["key"].lower() in user_text.lower():
        #         matches.append(f"{r['key']} → {r['value']} (last updated {r['last_updated']})")

        # if matches:
        #     return "Here’s what I remember:\n" + "\n".join(matches)

        # return "I couldn’t find anything in memory that matches."
    
    def add_event_from_natural(self, text: str) -> str:
        """
        Example: 'Add a weekly standup every Monday at 10am starting October 6th'
        """
        # Extract title
        title_match = re.search(r"(?:add|schedule) (.+?) (every|weekly|daily|monthly)", text.lower())
        title = title_match.group(1).title() if title_match else "Untitled Event"

        # Extract recurrence phrase
        recur_match = re.search(r"(every .+|daily .+|weekly .+|monthly .+)", text.lower())
        phrase = recur_match.group(1) if recur_match else None

        # Extract start date/time
        start_match = re.search(r"(starting|on|beginning) (.+)", text.lower())
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
            return f"Recurring event '{title}' created (id={event_id})."
        else:
            return "I couldn’t detect the recurrence pattern (e.g. 'every Monday at 10am')."
    
    def control_device_from_natural(self, text: str) -> str:
        """
        Stub for smart home commands like:
        - 'Turn on the kitchen lights'
        - 'Set the thermostat to 72 degrees'
        """
        if "light" in text.lower():
            if "off" in text.lower():
                return "Okay, I’ve turned off the lights."
            elif "on" in text.lower():
                return "Okay, I’ve turned on the lights."
            else:
                return "Should I turn the lights on or off?"

        if "thermostat" in text.lower() or "temperature" in text.lower():
            m = re.search(r"(\d{2})", text)
            if m:
                return f"Okay, thermostat set to {m.group(1)}°."
            return "What temperature should I set?"

        return "Device control not yet implemented for that command."
        
    def query_kg_context(self, user_text: str) -> str:
        tokens = user_text.lower().split()
        now = datetime.utcnow().isoformat()

        ask_past = any(phrase in user_text.lower() for phrase in [
            "used to", "was my", "were my", "used be", "formerly", "in the past"
        ])
        ask_future = any(phrase in user_text.lower() for phrase in [
        "will", "next", "in", "upcoming", "future"
        ])

        # detect explicit time reference
        time_start, time_end = extract_time_from_text(user_text)

        # Event questions (planning)
        if any(x in tokens for x in ["upcoming", "schedule", "meetings", "agenda", "calendar", "next", "week", "month"]):
            # choose a reasonable window (e.g., 14 days) or parse with your time parser
            from ..utils.timeparse import extract_time_from_text
            time_start, time_end = extract_time_from_text(user_text)
            if time_start and time_end:
                events = self.calendar.expand(time_start, time_end)
            else:
                events = self.query_upcoming_events(window_days=14)

            if events:
                lines = []
                for ev in events[:20]:
                    lines.append(f"{ev['start']} – {ev['title']}" + (f" @ {ev['location']}" if ev.get('location') else ""))
                return "Upcoming Events:\n" + "\n".join(lines)
            
            
        # detect future queries
        if ask_future:
            words = user_text.split()
            candidates = [w for w in words if w.istitle()]
            if candidates:
                entity = candidates[-1]
                rels = self.kg_store.query_future_relations(entity)
                if rels:
                    facts = []
                    for src, rel, tgt, vfrom, vto in rels:
                        facts.append(
                            f"{src} will {rel.replace('_',' ')} {tgt} "
                            f"(starting {vfrom}{' until ' + vto if vto else ''})"
                        )
                    return "Knowledge Graph Future Facts:\n" + "\n".join(facts)
                
        if any(t in tokens for t in [
            "who", "relation", "related", "about", "husband", "wife",
            "parent", "child", "boss", "mom", "dad", "work", "job", "sleep"
        ]):
            words = user_text.split()
            candidates = [w for w in words if w.istitle()]
            if candidates:
                entity = candidates[-1]

                if ask_past or time_start:
                    # fetch relations valid during specified time
                    rels = self.kg_store.query_relations(entity, at_time=time_start or now)
                    if rels:
                        facts = []
                        for src, rel, tgt, conf, vfrom, vto in rels:
                            # check if relation valid in requested time
                            if time_start and time_end:
                                if vfrom > time_end or (vto and vto < time_start):
                                    continue
                            tense = "was" if vto and vto < now else "is"
                            facts.append(
                                f"{src} {tense} {rel.replace('_',' ')} {tgt} "
                                f"(from {vfrom} until {vto or 'present'})"
                            )
                        if facts:
                            return "Knowledge Graph Time-Bounded Facts:\n" + "\n".join(facts)
                        else:
                            return "No facts found for that time frame."

                # normal active reasoning
                paths = self.kg_store.multi_hop(entity, max_hops=3, direction="both", at_time=now)
                formatted = []
                for path in paths:
                    pieces = []
                    for src, rel, tgt, conf, vfrom, vto in path:
                        tense = "is" if not vto or vto >= now else "was"
                        pieces.append(f"{src} {tense} {rel.replace('_',' ')} {tgt}")
                    formatted.append(" → ".join(pieces))
                if formatted:
                    return "Knowledge Graph Reasoning:\n" + "\n".join(formatted)

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
        """
        Handles 'forget that Alice likes pizza' or 'erase my old address'.
        """
        words = user_text.split()
        target = None
        for w in words:
            if w.istitle():  # naive entity grab
                target = w
                break

        if not target:
            return "What should I forget?"

        self.db.conn.execute("DELETE FROM facts WHERE key LIKE ?", (f"%{target}%",))
        self.db.conn.commit()
        return f"I’ve forgotten what I knew about {target}."

    def chat_brain(self, text: str) -> str:
        """
        Fallback to ChatGPT when no module matches.
        """
        try:
            resp = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are Ultron, a helpful assistant."},
                    {"role": "user", "content": text},
                ]
            )
            return resp["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return f"[ChatGPT Fallback Error] {e}"  

    def shutdown(self):
        self.scheduler.stop()
        self.db.close()