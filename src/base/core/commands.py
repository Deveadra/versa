# assistant/base/core/commands.py
from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import UTC, datetime, timedelta

from base.agents.orchestrator import Orchestrator
from base.core.nlu import parse_diagnostic_intent
from base.policy.audit_reader import recent_audits
from base.policy.policy_store import PolicyStore

try:
    from base.core.nlu import parse_diagnostic_intent
except Exception:  # pragma: no cover
    parse_diagnostic_intent = None  # type: ignore

STOP_HARD = re.compile(
    r"\b(stop|never|don'?t.*ever)\b.*\b(remind|bring up)\b.*\b(?P<topic>\w[\w\s-]{1,40})", re.I
)
PAUSE_SOFT = re.compile(
    r"\b(pause|stop)\b.*\b(?P<topic>\w[\w\s-]{1,40})(?:\b.*\bfor\s+(?P<num>\d+)\s*(?P<unit>day|days|week|weeks))?",
    re.I,
)
RESUME = re.compile(r"\b(resume|re-enable|start)\b.*\b(?P<topic>\w[\w\s-]{1,40})", re.I)


def normalize_topic(s: str) -> str:
    return re.sub(r"[^a-z0-9_-]+", "_", s.strip().lower())

def handle_policy_command(text: str, policy: PolicyStore) -> str | None:
    m = STOP_HARD.search(text)
    if m:
        topic = normalize_topic(m.group("topic"))
        policy.set_override(topic, "hard", reason="user_hard_stop")
        return f"Got it. I won’t bring up {topic} again unless you re-enable it."

    m = PAUSE_SOFT.search(text)
    t = text.lower().strip()
    
    # m = PAUSE_SOFT.search(text)
    if m:
        topic = normalize_topic(m.group("topic"))
        num = int(m.group("num") or 1)
        unit = m.group("unit") or "days"
        unit = unit.lower()
        days = num if unit.startswith("day") else num * 7
        expires = datetime.now(UTC) + timedelta(days=days)
        policy.set_override(topic, "soft", reason="user_soft_pause", expires_at=expires)
        return f"Got it. I’ll pause mentioning {topic} for {num} {unit}."

    m = RESUME.search(text)
    if m:
        topic = normalize_topic(m.group("topic"))
        policy.clear_overrides(topic)
        return f"I’ve re-enabled talking about {topic}."


    # === List active rules ===
    if re.search(r"\blist (my )?(rules|engagement rules)\b", t):
        rows = policy.conn.execute(
            """
            SELECT name, topic_id, priority, enabled
            FROM engagement_rules
            ORDER BY priority ASC
            LIMIT 20
        """
        ).fetchall()
        if not rows:
            return "I have no active rules."
        lines = [
            f"- {r['name']} (topic={r['topic_id']}, priority={r['priority']}, {'enabled' if r['enabled'] else 'disabled'})"
            for r in rows
        ]
        return "Here are my current rules:\n" + "\n".join(lines)

    # === Disable a rule ===
    m = re.search(r"\bdisable (rule )?(?P<name>[\w\-_]+)\b", t)
    if m:
        name = m.group("name")
        policy.conn.execute("UPDATE engagement_rules SET enabled=0 WHERE name=?", (name,))
        policy.conn.commit()
        return f"I’ve disabled rule '{name}'."

    # === Enable a rule ===
    m = re.search(r"\benable (rule )?(?P<name>[\w\-_]+)\b", t)
    if m:
        name = m.group("name")
        policy.conn.execute("UPDATE engagement_rules SET enabled=1 WHERE name=?", (name,))
        policy.conn.commit()
        return f"I’ve enabled rule '{name}'."

    # === Show audits (last night’s changes) ===
    if re.search(r"\b(show|what|tell me).*(audits|changes|last night)\b", t):
        audits = recent_audits(policy.conn, limit=5)
        if not audits:
            return "I didn’t make any changes recently."
        lines = [f"{a['created_at']}: {a['rationale']}" for a in audits]
        return "Here’s what I adjusted:\n" + "\n".join(lines)

    return None

def handle_diagnostic_command(text: str) -> str | None:
    """
    Natural language → robust diagnostic execution via Orchestrator.
    Primary path: Orchestrator.run_diagnostic (full pipeline).
    Fallback: run scripts/diagnostic_scan.py directly.

    Examples:
      - "Ultron, run a diagnostic scan"
      - "Ultron, quick diagnostic"
      - "Ultron, full diagnostic"
      - "Ultron, scan and fix"
      - "Ultron, optimize yourself"
    """
    t = (text or "").strip()
    if not t:
        return None
    low = t.lower()

    # ---- Intent detection (NLU first, keyword fallback) ----
    mode = "changed"
    fix = False

    intent = None
    if callable(parse_diagnostic_intent):
        try:
            intent = parse_diagnostic_intent(t)  # {"name":"diagnostic","mode":"all|changed","fix":bool,...}
        except Exception:
            intent = None

    if intent:
        mode = intent.get("mode", "changed")
        fix = bool(intent.get("fix", False))
    else:
        # lightweight keyword fallback
        if not any(k in low for k in ("diagnostic", "diagnostics", "scan", "laggy", "health check")):
            return None
        mode = "all" if any(k in low for k in ("all", "full", "entire", "everything", "deep")) else "changed"
        fix = any(k in low for k in ("fix", "optimize", "autofix", "auto-fix", "cleanup", "clean up", "format"))

    # ---- Execute: orchestrator first, script fallback ----
    orch = Orchestrator()
    try:
        return orch.run_diagnostic(mode=mode, fix=fix)
    except Exception as e_orch:
        try:
            args = [sys.executable, "scripts/diagnostic_scan.py", f"--{mode}"]
            if fix:
                args.append("--fix")
            result = subprocess.run(args, check=False, capture_output=True, text=True)
            output = (result.stdout or result.stderr or "").strip()
            return f"Diagnostics completed.\n{output}" if output else "Diagnostics completed."
        except Exception as e_script:
            return f"⚠️ Failed to run diagnostics: orchestrator={e_orch} | script={e_script}"

def handle_diagnostic_history(text: str, store) -> str | None:
    """
    Natural language → recall past diagnostic runs.
    Examples:
      - "show diagnostic history"
      - "what were the last diagnostic results?"
      - "why were you laggy yesterday?"
    """
    t = text.lower().strip()
    if "diagnostic" in t and any(k in t for k in ("history", "recent", "last")):
        items = store.recent_diagnostics(limit=5)
        if not items:
            return "I don’t have any recorded diagnostics yet."

        lines = []
        for ev in items:
            when = ev.get("started_at") or ev.get("created_at")
            mode = ev.get("mode")
            fix = "fix" if ev.get("fix") else "check"
            lag = "laggy" if ev.get("laggy") else "ok"
            cnt = len(ev.get("issues", []))
            lines.append(f"- {when} • {mode}/{fix} • {lag} • {cnt} issue(s)")
        return "Recent diagnostics:\n" + "\n".join(lines)

    # Trigger patterns
    history_trigger = (
        (re.search(r"\b(diagnostic|diagnostics|scan)\b", t) and
         re.search(r"\b(history|recent|last|previous|yesterday|earlier|report|results|summary)\b", t))
        or re.search(r"(what did.*(find|see|discover)|why.*lag|what.*results?)", t)
    )
    if not history_trigger:
        return None

    orch = Orchestrator()
    store = getattr(orch, "store", None)
    if store is None:
        return "I don’t have a diagnostic log available."

    # Optional time filter
    since_iso = None
    if "yesterday" in t or "last night" in t:
        since_iso = (datetime.now(UTC) - timedelta(days=1)).isoformat()

    # Fetch events with graceful fallbacks
    try:
        if since_iso and hasattr(store, "diagnostics_since"):
            items = store.diagnostics_since(since_iso, limit=10)
        elif hasattr(store, "recent_diagnostics"):
            items = store.recent_diagnostics(limit=5)
        elif hasattr(store, "list_events"):
            rows = store.list_events(type_="diagnostic", limit=5)
            items = []
            for r in rows:
                try:
                    items.append(json.loads(r.get("content", "{}")))
                except Exception:
                    pass
        else:
            items = []
    except Exception:
        items = []

    if not items:
        return "No recorded diagnostics yet."

    # If user asked for "last / latest"
    if re.search(r"\b(last|most recent|latest)\b", t) and hasattr(store, "last_diagnostic"):
        last = store.last_diagnostic()
        if last:
            return _format_last_diag(last)

    # Otherwise summarize recent
    lines = []
    for ev in items[:5]:
        when = ev.get("started_at") or ev.get("created_at") or "unknown time"
        mode = ev.get("mode", "?")
        fix = "fix" if ev.get("fix") else "check"
        lag = "laggy" if ev.get("laggy") else "ok"
        cnt = len(ev.get("issues", []))
        lines.append(f"- {when} • {mode}/{fix} • {lag} • {cnt} issue(s)")
    return "Recent diagnostics:\n" + "\n".join(lines)


def _format_last_diag(ev: dict) -> str:
    when = ev.get("started_at") or ev.get("created_at", "unknown time")
    mode = ev.get("mode", "?")
    fix = "fix" if ev.get("fix") else "check"
    lag = "laggy" if ev.get("laggy") else "ok"
    issues = ev.get("issues", [])
    preview = "; ".join(
        f"{i.get('file', '?')}: {i.get('summary', i.get('issue', '?'))}" for i in issues[:3]
    )
    more = "" if len(issues) <= 3 else f" …and {len(issues) - 3} more."
    return f"Last diagnostic ({when}): {mode}/{fix} • {lag} • {len(issues)} issues. {preview}{more}"
