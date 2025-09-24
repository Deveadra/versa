# assistant/base/policy/policy_store.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from loguru import logger
import sqlite3

POLICIES = ("principled", "advocate", "adaptive")
OVERRIDE_TYPES = ("hard", "soft", "preference")

@dataclass
class Topic:
    topic_id: str
    policy: str
    conviction: float = 0.75  # 0..1

class PolicyStore:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    # ---------- CRUD ----------
    def get_topic(self, topic_id: str) -> Optional[Topic]:
        row = self.conn.execute(
            "SELECT topic_id, policy, conviction FROM topics WHERE topic_id=?",
            (topic_id,),
        ).fetchone()
        if not row:
            return None
        return Topic(topic_id=row["topic_id"], policy=row["policy"], conviction=row["conviction"])

    def upsert_topic(self, topic_id: str, policy: str, conviction: float = 0.75):
        assert policy in POLICIES
        self.conn.execute(
            """
            INSERT INTO topics(topic_id, policy, conviction, created_at)
            VALUES(?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(topic_id) DO UPDATE SET
                policy=excluded.policy,
                conviction=excluded.conviction
            """,
            (topic_id, policy, conviction),
        )
        self.conn.commit()

    def set_override(self, topic_id: str, type_: str, reason: Optional[str] = None, expires_at: Optional[datetime] = None):
        assert type_ in OVERRIDE_TYPES
        self.conn.execute(
            "INSERT INTO topic_overrides(topic_id, type, reason, expires_at, created_at) VALUES(?,?,?,?,CURRENT_TIMESTAMP)",
            (topic_id, type_, reason, expires_at.isoformat() if expires_at else None),
        )
        self.conn.commit()

    def clear_overrides(self, topic_id: str):
        self.conn.execute("DELETE FROM topic_overrides WHERE topic_id=?", (topic_id,))
        self.conn.commit()

    def latest_override(self, topic_id: str) -> Optional[dict]:
        row = self.conn.execute(
            "SELECT * FROM topic_overrides WHERE topic_id=? ORDER BY id DESC LIMIT 1",
            (topic_id,),
        ).fetchone()
        return dict(row) if row else None

    def record_feedback(self, topic_id: str, feedback: str):
        # feedback: "acted" | "thanks" | "ignore" | "angry"
        self.conn.execute(
            "INSERT INTO topic_feedback(topic_id, feedback, created_at) VALUES(?,?,CURRENT_TIMESTAMP)",
            (topic_id, feedback),
        )
        self.conn.commit()

    def bump_ignore(self, topic_id: str):
        self.conn.execute(
            """
            INSERT INTO topic_state(topic_id, ignore_count, escalation_count, last_mentioned)
            VALUES(?, 1, 0, CURRENT_TIMESTAMP)
            ON CONFLICT(topic_id) DO UPDATE SET
                ignore_count = topic_state.ignore_count + 1,
                last_mentioned = CURRENT_TIMESTAMP
            """,
            (topic_id,),
        )
        self.conn.commit()

    def record_mention(self, topic_id: str, escalated: bool):
        self.conn.execute(
            """
            INSERT INTO topic_state(topic_id, ignore_count, escalation_count, last_mentioned)
            VALUES(?, 0, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(topic_id) DO UPDATE SET
                last_mentioned = CURRENT_TIMESTAMP,
                escalation_count = CASE WHEN ? THEN topic_state.escalation_count + 1 ELSE topic_state.escalation_count END
            """,
            (topic_id, 1 if escalated else 0, 1 if escalated else 0),
        )
        self.conn.commit()

    def state(self, topic_id: str) -> dict:
        row = self.conn.execute(
            "SELECT * FROM topic_state WHERE topic_id=?", (topic_id,)
        ).fetchone()
        return dict(row) if row else {"topic_id": topic_id, "ignore_count": 0, "escalation_count": 0, "last_mentioned": None}

    # ---------- evaluation ----------
    def should_speak(self, topic_id: str, context_signals: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Decide whether to speak on a topic now, with what tone.
        """
        topic = self.get_topic(topic_id)
        if not topic:
            # default new topics to advocate (safer than adaptive) unless you prefer principled defaults
            topic = Topic(topic_id=topic_id, policy="advocate", conviction=0.75)
            self.upsert_topic(topic_id, topic.policy, topic.conviction)

        override = self.latest_override(topic_id)
        now = datetime.utcnow()

        # hard override blocks completely
        if override and override["type"] == "hard":
            return False, {"reason": "hard_override"}

        # honor soft override while active; allow trigger- or time-based reintro
        if override and override["type"] == "soft":
            exp = override.get("expires_at")
            if exp:
                try:
                    if now < datetime.fromisoformat(exp):
                        if not context_signals.get("trigger", False):
                            return False, {"reason": "soft_override_active"}
                except Exception:
                    pass  # malformed timestamp -> treat as expired
            # else expired -> proceed

        st = self.state(topic_id)
        ignore_penalty_cap = {"principled": 0.25, "advocate": 0.5, "adaptive": 1.0}[topic.policy]
        ignore_penalty = min(st.get("ignore_count", 0) * 0.05, ignore_penalty_cap)

        # base conviction (0..1), boosted by context relevance
        ctx = 0.0
        if context_signals.get("long_sitting"): ctx += 0.5
        if context_signals.get("approaching_bedtime"): ctx += 0.6
        if context_signals.get("health_risk"): ctx += 0.7
        if context_signals.get("time_window_bonus"): ctx += 0.3

        # recent feedback nudge
        fb_row = self.conn.execute(
            "SELECT feedback FROM topic_feedback WHERE topic_id=? ORDER BY id DESC LIMIT 1",
            (topic_id,),
        ).fetchone()
        fb_adj = 0.0
        if fb_row:
            fb = fb_row["feedback"]
            fb_adj = {"acted": +0.2, "thanks": +0.1, "ignore": -0.1, "angry": -0.2}.get(fb, 0.0)

        # policy threshold
        thresholds = {"principled": 0.35, "advocate": 0.5, "adaptive": 0.7}
        score = topic.conviction + ctx + fb_adj - ignore_penalty
        speak = score >= thresholds[topic.policy]

        tone = "gentle"
        if speak:
            if score >= thresholds[topic.policy] + 0.35:
                tone = "firm"
            elif score >= thresholds[topic.policy] + 0.15:
                tone = "persistent"

        return speak, {"tone": tone, "score": round(score, 3), "policy": topic.policy}
