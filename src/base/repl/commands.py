# src/base/repl/commands.py
from __future__ import annotations

import inspect
import re
from typing import Any, Callable

from base.core.commands import (
    handle_diagnostic_command,
    handle_diagnostic_history,
    handle_policy_command,
)

from ..agents.orchestrator import Orchestrator

_orch: Orchestrator | None = None


def _get_orch() -> Orchestrator:
    global _orch
    if _orch is None:
        _orch = Orchestrator()
    return _orch


def _call_text_with_optional_store(fn: Any, text: str, store: Any) -> Any:
    """
    Call either fn(text) or fn(text, store) depending on its signature.
    Uses positional args only to avoid 'No parameter named store' type errors.
    """
    try:
        params = inspect.signature(fn).parameters
        if "store" in params or len(params) >= 2:
            return fn(text, store)
        return fn(text)
    except Exception:
        return fn(text)


def _call_policy_with_optional_store(fn: Any, text: str, policy: Any, store: Any) -> Any:
    """
    Call either fn(text, policy) or fn(text, policy, store) depending on signature.
    Positional only to stay Pylance-safe.
    """
    try:
        params = inspect.signature(fn).parameters
        if "store" in params or len(params) >= 3:
            return fn(text, policy, store)
        return fn(text, policy)
    except Exception:
        return fn(text, policy)


def handle_command(cmd: str, text: str, policy) -> str | None:
    orch = _get_orch()
    tokens = cmd.strip().split(maxsplit=2)

    # 1) Policy rules
    resp = _call_policy_with_optional_store(handle_policy_command, text, policy, orch.store)
    if resp:
        return resp

    # 2) Run diagnostics
    resp = _call_text_with_optional_store(handle_diagnostic_command, text, orch.store)
    if resp:
        return resp

    # 3) Read diagnostic history/summaries
    resp = _call_text_with_optional_store(handle_diagnostic_history, text, orch.store)
    if resp:
        return resp

    # (future: add other handlers here...)

    if not tokens:
        return ""

    if tokens[0] == "event":
        if len(tokens) < 2:
            return "Usage: event add|list|upcoming"

        sub = tokens[1]

        if sub == "add":
            match = re.match(r'recurring "(.+)" (.+) start ([^ ]+)', cmd, re.IGNORECASE)
            if match:
                title = match.group(1)
                phrase = match.group(2)
                start_iso = match.group(3)
                event_id = orch.create_recurring_event_from_phrase(title, phrase, start_iso)
                return f"Recurring event '{title}' created (id={event_id})."
            return 'Usage: event add recurring "Title" <phrase> start <ISO date>'

        if sub == "list":
            events = orch.calendar.list_events()
            if not events:
                return "No events stored."
            return "\n".join(
                f"[{e['id']}] {e['title']} start={e['start']} rrule={e['rrule'] or '—'}"
                for e in events
            )

        if sub == "upcoming":
            days = int(tokens[2]) if len(tokens) > 2 and tokens[2].isdigit() else 14
            evs = orch.query_upcoming_events(window_days=days)
            if not evs:
                return f"No events in the next {days} days."
            return "\n".join(
                f"{e['start']} – {e['title']}" + (f" @ {e['location']}" if e.get("location") else "")
                for e in evs
            )

    return None
