
from __future__ import annotations

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
        reply = self.brain.complete(SYSTEM_PROMPT, prompt)
        return reply
            
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

    

    def shutdown(self):
        self.scheduler.stop()
        self.db.close()