# assistant/base/core/commands.py
from __future__ import annotations
import re
from datetime import datetime, timedelta
from typing import Optional, Tuple
from base.policy.policy_store import PolicyStore

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
    if m:
        topic = normalize_topic(m.group("topic"))
        num = int(m.group("num"))
        unit = m.group("unit").lower()
        days = num * (7 if unit.startswith("week") else 1)
        policy.set_override(topic, "soft", reason="user_soft_pause", expires_at=datetime.utcnow() + timedelta(days=days))
        return f"Okay. I’ll pause {topic} for {days} days."

    m = RESUME.search(text)
    if m:
        topic = normalize_topic(m.group("topic"))
        policy.clear_overrides(topic)
        return f"I’ll start mentioning {topic} again."

    return None
