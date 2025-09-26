# assistant/base/core/commands.py
from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Optional, Tuple
from base.policy.policy_store import PolicyStore
from base.policy.audit_reader import recent_audits


STOP_HARD = re.compile(r"\b(stop|never|don'?t.*ever)\b.*\b(remind|bring up)\b.*\b(?P<topic>\w[\w\s-]{1,40})", re.I)
PAUSE_SOFT = re.compile(r"\b(pause|stop)\b.*\b(?P<topic>\w[\w\s-]{1,40})\b.*\bfor\s+(?P<num>\d+)\s*(?P<unit>day|days|week|weeks)", re.I)
RESUME = re.compile(r"\b(resume|re-enable|start)\b.*\b(?P<topic>\w[\w\s-]{1,40})", re.I)

def normalize_topic(s: str) -> str:
    return re.sub(r"[^a-z0-9_-]+", "_", s.strip().lower())

def handle_policy_command(text: str, policy: PolicyStore) -> Optional[str]:
    m = STOP_HARD.search(text)
    if m:
        topic = normalize_topic(m.group("topic"))
        policy.set_override(topic, "hard", reason="user_hard_stop")
        return f"Got it. I won’t bring up {topic} again unless you re-enable it."

    m = PAUSE_SOFT.search(text)
    t = text.lower().strip()

    # === List active rules ===
    if re.search(r"\blist (my )?(rules|engagement rules)\b", t):
        rows = policy.conn.execute("""
            SELECT name, topic_id, priority, enabled
            FROM engagement_rules
            ORDER BY priority ASC
            LIMIT 20
        """).fetchall()
        if not rows:
            return "I have no active rules."
        lines = [f"- {r['name']} (topic={r['topic_id']}, priority={r['priority']}, {'enabled' if r['enabled'] else 'disabled'})"
                 for r in rows]
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