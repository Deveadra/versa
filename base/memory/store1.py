
# base/memory/store.py
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from typing import Iterable, List, Tuple, Any, Optional

from loguru import logger

# your project imports
from config.config import settings
from base.database.sqlite import SQLiteConn           # ✅ correct path
from .scoring import assess_importance

class MemoryStore:
    """
    Lightweight event/fact store on SQLite.
    Accepts either an SQLiteConn wrapper (with .conn) or a raw sqlite3.Connection.
    """

    def __init__(self, db: SQLiteConn | sqlite3.Connection):
        # Normalize to a raw sqlite3 connection
        self.conn: sqlite3.Connection = db.conn if hasattr(db, "conn") else db  # type: ignore[attr-defined]
        self.conn.row_factory = sqlite3.Row
        self._fts_enabled = False
        self._ensure_schema()

    # ---------- schema ----------
    def _ensure_schema(self) -> None:
        cur = self.conn.cursor()

        # Simple key/value facts
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS facts (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                last_updated TEXT NOT NULL
            )
            """
        )

        # Events (memories)
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                ts TEXT NOT NULL,
                importance REAL NOT NULL DEFAULT 0.0,
                type TEXT NOT NULL DEFAULT 'event'
            )
            """
        )

        # Optional FTS (not always compiled on Windows Python)
        try:
            cur.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS events_fts
                USING fts5(content, content='events', content_rowid='id')
                """
            )
            self._fts_enabled = True
        except sqlite3.OperationalError:
            # FTS not available; we’ll fall back to LIKE queries
            self._fts_enabled = False

        self.conn.commit()

    # ---------- facts ----------
    def upsert_fact(self, key: str, value: str) -> None:
        ts = datetime.utcnow().isoformat()
        self.conn.execute(
            """
            INSERT INTO facts(key, value, last_updated)
            VALUES(?, ?, ?)
            ON CONFLICT(key)
            DO UPDATE SET value=excluded.value, last_updated=excluded.last_updated
            """,
            (key, value, ts),
        )
        self.conn.commit()

    def list_facts(self) -> List[Tuple[str, str]]:
        cur = self.conn.execute("SELECT key, value FROM facts ORDER BY key")
        return [(r["key"], r["value"]) for r in cur.fetchall()]

    # Delete facts containing topic, and events whose content matches topic.
    def forget(self, topic: str) -> int:
        n1 = self.conn.execute(
            "DELETE FROM facts WHERE key LIKE ? OR value LIKE ?",
            (f"%{topic}%", f"%{topic}%"),
        ).rowcount or 0
        n2 = self.conn.execute(
            "DELETE FROM events WHERE content LIKE ?",
            (f"%{topic}%",),
        ).rowcount or 0
        self.conn.commit()
        return n1 + n2

    # EVENTS
    def add_event(self, content: str, importance: float = 0.0, type_: str = "event") -> int:
        ts = datetime.utcnow().isoformat()
        cur = self.conn.execute(
            "INSERT INTO events(content, ts, importance, type) VALUES(?, ?, ?, ?)",
            (content, ts, importance, type_),
        )
        rowid = cur.lastrowid
        if self._fts_enabled:
            try:
                self.conn.execute(
                    "INSERT INTO events_fts(rowid, content) VALUES(?, ?)",
                    (rowid, content),
                )
            except sqlite3.OperationalError:
                # If FTS table isn't available after all, just ignore
                self._fts_enabled = False
        self.conn.commit()
        rowid = cur.lastrowid
        if rowid is None:
            raise RuntimeError("Failed to retrieve lastrowid after insert")
        return rowid

    def maybe_store_text(self, text: str, explicit_type: Optional[str] = None) -> bool:
        score = assess_importance(text)
        if score < settings.importance_threshold:
            return False
        self.add_event(text, importance=score, type_=explicit_type or "event")
        return True

    def prune_events(self) -> int:
        ttl = timedelta(days=settings.memory_ttl_days)
        cutoff = (datetime.utcnow() - ttl).isoformat()
        cur = self.conn.execute(
            "DELETE FROM events WHERE ts < ? AND importance < ?",
            (cutoff, float(settings.importance_threshold)),
        )
        self.conn.commit()
        return int(cur.rowcount or 0)

    # ---------- retrieval ----------
    def keyword_search(self, query: str, limit: int = 5) -> List[str]:
        if self._fts_enabled:
            cur = self.conn.execute(
                "SELECT content FROM events_fts WHERE events_fts MATCH ? LIMIT ?",
                (query, limit),
            )
            return [r[0] for r in cur.fetchall()]
        # fallback: LIKE
        cur = self.conn.execute(
            "SELECT content FROM events WHERE content LIKE ? ORDER BY id DESC LIMIT ?",
            (f"%{query}%", limit),
        )
        return [r[0] for r in cur.fetchall()]
