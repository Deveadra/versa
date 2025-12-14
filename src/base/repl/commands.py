# base/repl/commands.py
import re

from base.core.commands import (
    handle_diagnostic_command,
    handle_diagnostic_history,
    handle_policy_command,
)

from ..agents.orchestrator import Orchestrator

orch = Orchestrator()


def handle_command(cmd: str, text: str, policy) -> str | None:
    tokens = cmd.strip().split(maxsplit=2)

    # 1) Policy rules
    resp = handle_policy_command(text, policy)
    if resp:
        return resp

    # 2) Run diagnostics
    resp = handle_diagnostic_command(text)
    if resp:
        return resp

    # 3) Read diagnostic history/summaries
    resp = handle_diagnostic_history(text)
    if resp:
        return resp

    # (future: add other handlers here...)

    # return None

    if not tokens:
        return ""

    if tokens[0] == "event":
        if len(tokens) < 2:
            return "Usage: event add|list|upcoming"

        sub = tokens[1]

        # event add recurring "Title" every monday at 10am start 2025-10-06T10:00:00Z
        if sub == "add":
            match = re.match(r'recurring "(.+)" (.+) start ([^ ]+)', cmd, re.IGNORECASE)
            if match:
                title = match.group(1)
                phrase = match.group(2)
                start_iso = match.group(3)
                event_id = orch.create_recurring_event_from_phrase(title, phrase, start_iso)
                return f"Recurring event '{title}' created (id={event_id})."
            return 'Usage: event add recurring "Title" <phrase> start <ISO date>'

        # event list (all events stored, without expansion)
        if sub == "list":
            events = orch.calendar.list_events()
            if not events:
                return "No events stored."
            return "\n".join(
                f"[{e['id']}] {e['title']} start={e['start']} rrule={e['rrule'] or '—'}"
                for e in events
            )

        # event upcoming 14
        if sub == "upcoming":
            days = int(tokens[2]) if len(tokens) > 2 and tokens[2].isdigit() else 14
            evs = orch.query_upcoming_events(window_days=days)
            if not evs:
                return f"No events in the next {days} days."
            return "\n".join(
                f"{e['start']} – {e['title']}"
                + (f" @ {e['location']}" if e.get("location") else "")
                for e in evs
            )

    return "" or None


