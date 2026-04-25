# src/base/repl/commands.py
from __future__ import annotations

import inspect
import re
from typing import Any

from base.agents.orchestrator import Orchestrator
from base.core.commands import (
    handle_diagnostic_command,
    handle_diagnostic_history,
    handle_policy_command,
)
from base.self_improve.self_improve_db import (
    ensure_self_improve_schema,
    fetch_open_gaps,
)

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
                f"{e['start']} – {e['title']}"
                + (f" @ {e['location']}" if e.get("location") else "")
                for e in evs
            )

        # 4) Dream / self-improve status
    if tokens and tokens[0] in {"dream", "selfimprove", "self-improve"}:
        if len(tokens) < 2:
            return "Usage: dream status | dream gaps [limit]"

        sub = tokens[1].lower()

        # Always ensure schema exists (no-op if already migrated)
        try:
            ensure_self_improve_schema(orch.db.conn)
        except Exception as e:
            return f"Self-improve schema check failed: {e}"

        if sub == "status":
            conn = orch.db.conn

            # Latest score run
            cur = conn.execute(
                """
                SELECT id, created_at, run_type, mode, fix_enabled, git_branch, git_sha, score, passed
                FROM repo_score_runs
                ORDER BY id DESC
                LIMIT 1
                """
            )
            r = cur.fetchone()
            if not r:
                return "No self-improve runs recorded yet."

            # sqlite3.Row supports both index and key access; use index to be safe.
            run_id = r[0]
            created_at = r[1]
            run_type = r[2]
            mode = r[3]
            fix_enabled = bool(r[4])
            git_branch = r[5] or "—"
            git_sha = (r[6] or "—")[:10]
            score = float(r[7] or 0.0)
            passed = bool(r[8])

            # Latest improvement attempt (if any)
            cur2 = conn.execute(
                """
                SELECT id, created_at, iteration, branch, proposal_title, pr_url, improved, error_text
                FROM repo_improvement_attempts
                ORDER BY id DESC
                LIMIT 1
                """
            )
            a = cur2.fetchone()

            # Open gaps count
            cur3 = conn.execute(
                """
                SELECT COUNT(*)
                FROM capability_gaps
                WHERE status IN ('queued', 'in_progress', 'new')
                """
            )
            open_gaps = int(cur3.fetchone()[0])

            lines = []
            lines.append("Self-improve status:")
            lines.append(
                f"- Last score run: id={run_id} at {created_at} | type={run_type} mode={mode} "
                f"fix={'on' if fix_enabled else 'off'} passed={'yes' if passed else 'no'} "
                f"score={score:.3f} branch={git_branch} sha={git_sha}"
            )

            if a:
                attempt_id = a[0]
                attempt_at = a[1]
                iteration = a[2]
                branch = a[3]
                title = a[4] or "—"
                pr_url = a[5] or "—"
                improved = bool(a[6])
                error_text = (a[7] or "").strip()

                lines.append(
                    f"- Last attempt: id={attempt_id} at {attempt_at} | it={iteration} "
                    f"improved={'yes' if improved else 'no'} branch={branch} title={title}"
                )
                if pr_url != "—":
                    lines.append(f"  PR: {pr_url}")
                if error_text:
                    lines.append(f"  Error: {error_text}")
            else:
                lines.append("- Last attempt: —")

            lines.append(f"- Open gaps: {open_gaps} (run `dream gaps` to list)")

            return "\n".join(lines)

        if sub == "gaps":
            limit = 5
            if len(tokens) >= 3 and tokens[2].isdigit():
                limit = max(1, min(50, int(tokens[2])))

            gaps = fetch_open_gaps(orch.db.conn, limit=limit)
            if not gaps:
                return "No open gaps (queued/in_progress/new)."

            lines = [f"Open gaps (top {len(gaps)}):"]
            for g in gaps:
                lines.append(
                    f"- [#{g['id']}] prio={g['priority']} status={g['status']} "
                    f"source={g['source']} capability={g['requested_capability']}"
                )
                if g.get("observed_failure"):
                    lines.append(f"  failure: {g['observed_failure']}")
            return "\n".join(lines)

        return "Usage: dream status | dream gaps [limit]"

    return None
