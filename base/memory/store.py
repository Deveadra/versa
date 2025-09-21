
# base/memory/store.py
from __future__ import annotations

import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Tuple, Optional
from loguru import logger

from base.database.sqlite import SQLiteConn
from base.memory.scoring import assess_importance
from config.config import settings


DB_PATH = Path("memory.db")


class MemoryStore:
<<<<<<< HEAD
    def __init__(self, db: SQLiteConn):
        self.db = db
        
        
    def init_db():
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("""
=======
    """
    Persistent memory store using SQLite.
    Supports key/value facts, events with importance, and retrieval (FTS if available).
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
        
        # Memories (general)
        cur.execute(
            """
>>>>>>> d32adef7e91e5de16a6dd3e1b7ca2d053081636f
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                type TEXT,
                content TEXT,
                response TEXT
            )
<<<<<<< HEAD
        """)
        conn.commit()
        conn.close()

    def save_memory(self, memory: dict):
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO memories (timestamp, type, content, response)
            VALUES (?, ?, ?, ?)
        """, (memory["timestamp"], memory["type"], memory["content"], memory["response"]))
        conn.commit()
        conn.close()

    def fetch_recent(self, limit=10):
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT timestamp, type, content, response FROM memories ORDER BY id DESC LIMIT ?", (limit,))
        rows = cur.fetchall()
        conn.close()
        return rows

    def clear_old(self, days=30):
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("DELETE FROM memories WHERE timestamp <= datetime('now', ?)", (f'-{days} days',))
        conn.commit()
        conn.close()



    # MVP Section Added
    # FACTS
    def upsert_fact(self, key: str, value: str) -> None:
        ts = datetime.utcnow().isoformat()
        self.db.conn.execute(
            """
            INSERT INTO facts(key, value, last_updated)
            VALUES(?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value, last_updated=excluded.last_updated
            """,
            (key, value, ts),
        )
        self.db.conn.commit()

    # EVENTS
    def add_event(self, content: str, importance: float = 0.0, type_: str = "event") -> int:
        ts = datetime.utcnow().isoformat()
        cur = self.db.conn.execute(
            "INSERT INTO events(content, ts, importance, type) VALUES(?, ?, ?, ?)",
            (content, ts, importance, type_),
        )
        rowid = cur.lastrowid
        if rowid is None:
            raise RuntimeError("Failed to insert event: lastrowid is None")
        # FTS mirror (optional)
        self.db.conn.execute("INSERT INTO events_fts(rowid, content) VALUES(?, ?)", (rowid, content))
        self.db.conn.commit()
        return rowid

    def maybe_store_text(self, text: str, explicit_type: str | None = None) -> bool:
        score = assess_importance(text)
        if score < settings.importance_threshold:
            return False
        self.add_event(text, importance=score, type_=explicit_type or "event")
        return True

    # TTL / pruning
    def prune_events(self) -> int:
        ttl = timedelta(days=settings.memory_ttl_days)
        cutoff = (datetime.utcnow() - ttl).isoformat()
        cur = self.db.conn.execute("DELETE FROM events WHERE ts < ? AND importance < ?", (cutoff, settings.importance_threshold))
        self.db.conn.commit()
        return cur.rowcount
=======
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
            );
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

    def forget(self, topic: str) -> int:
        """Delete facts and events containing the given topic string."""
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

    # ---------- events ----------
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
        """Prune old, low-importance events according to TTL."""
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
>>>>>>> d32adef7e91e5de16a6dd3e1b7ca2d053081636f

    # Retrieval (keyword fallback)
    def keyword_search(self, query: str, limit: int = 5) -> List[str]:
        cur = self.db.conn.execute(
            "SELECT content FROM events_fts WHERE events_fts MATCH ? LIMIT ?",
            (query, limit),
        )
        return [r[0] for r in cur.fetchall()]

    def list_facts(self) -> list[tuple[str, str]]:
        return list(self.db.conn.execute("SELECT key, value FROM facts ORDER BY key"))

<<<<<<< HEAD
    def forget(self, topic: str) -> int:
        n1 = self.db.conn.execute("DELETE FROM facts WHERE key LIKE ? OR value LIKE ?", (f"%{topic}%", f"%{topic}%")).rowcount
        n2 = self.db.conn.execute("DELETE FROM events WHERE content LIKE ?", (f"%{topic}%",)).rowcount
        self.db.conn.commit()
        return n1 + n2
=======
# Outside of class for legacy compatibility

def _connect_for_compat() -> sqlite3.Connection:
    # use your main DB so everything stays in one file
    return sqlite3.connect(settings.db_path, check_same_thread=False)

def init_db() -> None:
    """
    Legacy initializer: create schema if needed.
    """
    with _connect_for_compat() as conn:
        # constructing MemoryStore ensures schema
        MemoryStore(conn)  # __init__ calls _ensure_schema()
        # nothing else to do

def save_memory(memory: dict) -> None:
    """
    Legacy saver used by decide_memory(). Expects keys:
      - timestamp (optional, ISO string)
      - type
      - content
      - response
    """
    ts = memory.get("timestamp") or datetime.utcnow().isoformat()
    typ = memory.get("type", "event")
    content = memory.get("content", "")
    response = memory.get("response", "")

    with _connect_for_compat() as conn:
        conn.execute(
            "INSERT INTO memories(timestamp, type, content, response) VALUES(?, ?, ?, ?)",
            (ts, typ, content, response),
        )
        conn.commit()
>>>>>>> d32adef7e91e5de16a6dd3e1b7ca2d053081636f
