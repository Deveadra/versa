from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any
import json

from loguru import logger
from database.sqlite import SQLiteConn

@dataclass
class UsageEvent:
    user_text: Optional[str]
    normalized_intent: Optional[str]
    resolved_action: Optional[str]
    params: Dict[str, Any]
    success: Optional[bool]
    latency_ms: Optional[int]

class UsageLogger:
    def __init__(self, conn: SQLiteConn):
        self.conn = conn

    def log(self, ev: UsageEvent) -> int:
        c = self.conn.cursor()
        c.execute(
            """
            INSERT INTO usage_log (user_text, normalized_intent, resolved_action, params_json, success, latency_ms)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                ev.user_text,
                ev.normalized_intent,
                ev.resolved_action,
                json.dumps(ev.params or {}, ensure_ascii=False),
                1 if ev.success else 0 if ev.success is not None else None,
                ev.latency_ms,
            ),
        )
        self.conn.conn.commit()
        usage_id = int(c.lastrowid or 0)
        logger.debug(f"usage_log inserted id={usage_id}")
        return usage_id

    def recent(self, limit: int = 100):
        c = self.conn.cursor()
        c.execute("SELECT * FROM usage_log ORDER BY id DESC LIMIT ?", (limit,))
        return [dict(r) for r in c.fetchall()]
