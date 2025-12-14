# base/repl/nl_commands.py
import re
from datetime import datetime

from dateutil import parser as dateparser

from ..agents.orchestrator import Orchestrator

orch = Orchestrator()


def handle_nl_command(cmd: str) -> str:
    cmd_low = cmd.lower().strip()

    # -------------------
    # KG queries (natural)
    # -------------------
    # -------------------
    # MEMORY
    # -------------------
    if any(
        kw in cmd_low
        for kw in ["do you remember", "what did i tell you", "did i mention", "have you stored"]
    ):
        return orch.query_memory_context(cmd)

    if any(kw in cmd_low for kw in ["forget that", "erase", "remove memory of"]):
        return orch.forget_memory(cmd)

    # -------------------
    # KNOWLEDGE GRAPH
    # -------------------
    if any(
        phrase in cmd_low for phrase in ["used to", "was my", "were my", "formerly", "in the past"]
    ):
        return orch.query_kg_context(cmd)

    if re.search(r"(19|20)\d{2}", cmd_low) or any(
        kw in cmd_low
        for kw in [
            "last week",
            "next week",
            "last month",
            "next month",
            "yesterday",
            "tomorrow",
            "two weeks ago",
            "days ago",
            "next year",
            "last year",
        ]
    ):
        return orch.query_kg_context(cmd)

    if any(
        kw in cmd_low
        for kw in [
            "who",
            "where",
            "relation",
            "related",
            "boss",
            "parent",
            "child",
            "husband",
            "wife",
            "job",
            "work",
            "friend",
            "teacher",
            "attends",
            "sleep",
        ]
    ):
        return orch.query_kg_context(cmd)

    # -------------------
    # CALENDAR / EVENTS
    # -------------------
    if any(
        x in cmd_low
        for x in [
            "what's on my calendar",
            "my meetings",
            "my schedule",
            "agenda",
            "appointments",
            "events",
            "upcoming",
        ]
    ):
        # detect window
        days = 14
        if "next week" in cmd_low:
            days = 7
        elif "next month" in cmd_low:
            days = 30
        evs = orch.query_upcoming_events(window_days=days)
        if not evs:
            return f"No events found in the next {days} days."
        return "\n".join(
            f"{e['start']} – {e['title']}" + (f" @ {e['location']}" if e.get("location") else "")
            for e in evs
        )

    if any(x in cmd_low for x in ["list all events", "show all events"]):
        events = orch.calendar.list_events()
        if not events:
            return "No events stored."
        return "\n".join(
            f"[{e['id']}] {e['title']} start={e['start']} rrule={e['rrule'] or '—'}" for e in events
        )

    if cmd_low.startswith("add") or cmd_low.startswith("schedule"):
        return orch.add_event_from_natural(cmd)

    # -------------------
    # DEVICES (stub)
    # -------------------
    if any(
        x in cmd_low
        for x in ["turn on", "turn off", "switch on", "switch off", "dim", "set temperature"]
    ):
        return orch.control_device_from_natural(cmd)  # type: ignore[attr-defined]

    # -------------------
    # Past tense detection
    if any(
        phrase in cmd_low for phrase in ["used to", "was my", "were my", "formerly", "in the past"]
    ):
        return orch.query_kg_context(cmd)

    # Time-specific queries (years, months, relative times)
    if re.search(r"(19|20)\d{2}", cmd_low) or any(
        kw in cmd_low
        for kw in [
            "last week",
            "next week",
            "last month",
            "next month",
            "yesterday",
            "tomorrow",
            "two weeks ago",
            "days ago",
            "next year",
            "last year",
        ]
    ):
        return orch.query_kg_context(cmd)

    # General relation/entity queries
    if any(
        kw in cmd_low
        for kw in [
            "who",
            "where",
            "relation",
            "related",
            "boss",
            "parent",
            "child",
            "husband",
            "wife",
            "job",
            "work",
            "sleep",
            "friend",
            "teacher",
            "attends",
        ]
    ):
        return orch.query_kg_context(cmd)

    # --- Add recurring event ---
    # e.g. "add a weekly standup every monday at 10am starting October 6th"
    if cmd_low.startswith("add") or cmd_low.startswith("schedule"):
        # title extraction: everything after "add" until "every"/"weekly"/"daily"/"monthly"
        title_match = re.match(
            r"(?:add|schedule) (?:a |an )?(.+?) (every|weekly|daily|monthly)", cmd_low
        )
        if title_match:
            title = title_match.group(1).strip().title()
        else:
            title = "Untitled Event"

        # recurrence phrase
        recur_match = re.search(r"(every .+|daily .+|weekly .+|monthly .+)", cmd_low)
        phrase = recur_match.group(1) if recur_match else None

        # start date/time
        start_match = re.search(r"(starting|on|beginning) (.+)", cmd_low)
        if start_match:
            try:
                dt = dateparser.parse(start_match.group(2), fuzzy=True)
                start_iso = dt.isoformat()
            except Exception:
                return "Sorry, I couldn’t understand the start date."
        else:
            start_iso = datetime.utcnow().isoformat()

        if phrase:
            event_id = orch.create_recurring_event_from_phrase(title, phrase, start_iso)
            return f"Recurring event '{title}' created (id={event_id})."
        else:
            return "I couldn’t detect the recurrence pattern (e.g. 'every Monday at 10am')."

    # --- Upcoming events ---
    if any(
        x in cmd_low
        for x in ["what's on my calendar", "upcoming", "next week", "next month", "my schedule"]
    ):
        days = 14
        if "next week" in cmd_low:
            days = 7
        elif "next month" in cmd_low:
            days = 30
        evs = orch.query_upcoming_events(window_days=days)
        if not evs:
            return f"No events found in the next {days} days."
        return "\n".join(
            f"{e['start']} – {e['title']}" + (f" @ {e['location']}" if e.get("location") else "")
            for e in evs
        )

    # --- List all stored events (raw) ---
    if "list all events" in cmd_low or "show all events" in cmd_low:
        events = orch.calendar.list_events()
        if not events:
            return "No events stored."
        return "\n".join(
            f"[{e['id']}] {e['title']} start={e['start']} rrule={e['rrule'] or '—'}" for e in events
        )

    # -------------------
    # FALLBACK: ChatGPT brain
    # -------------------
    return orch.chat_brain(cmd)

    # return "Sorry, I didn’t understand that command."
