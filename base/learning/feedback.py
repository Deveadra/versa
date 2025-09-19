from __future__ import annotations
from typing import Optional
from database.sqlite import SQLiteConn

class Feedback:
    def __init__(self, conn: SQLiteConn):
        self.conn = conn

    def record(self, usage_id: int, kind: str, note: Optional[str] = None):
        c = self.conn.cursor()
        c.execute(
            "INSERT INTO feedback_events (usage_id, kind, note) VALUES (?, ?, ?)",
            (usage_id, kind, note),
        )
        self.conn.conn.commit()
