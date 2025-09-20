# base/memory/store.py
from __future__ import annotations
import sqlite3
import sqlite3

from pathlib import Path
from datetime import datetime, timedelta
from typing import Iterable, List, Tuple
from loguru import logger
from assistant.config import settings
from database.sqlite import SQLiteConn
from .scoring import assess_importance
from assistant.config.config import settings


DB_PATH = Path("memory.db")

class MemoryStore:
    def __init__(self, db: SQLiteConn):
        self.db = db
        
        
    def init_db():
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                type TEXT,
                content TEXT,
                response TEXT
            )
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

    # Retrieval (keyword fallback)
    def keyword_search(self, query: str, limit: int = 5) -> List[str]:
        cur = self.db.conn.execute(
            "SELECT content FROM events_fts WHERE events_fts MATCH ? LIMIT ?",
            (query, limit),
        )
        return [r[0] for r in cur.fetchall()]

    def list_facts(self) -> list[tuple[str, str]]:
        return list(self.db.conn.execute("SELECT key, value FROM facts ORDER BY key"))

    def forget(self, topic: str) -> int:
        n1 = self.db.conn.execute("DELETE FROM facts WHERE key LIKE ? OR value LIKE ?", (f"%{topic}%", f"%{topic}%")).rowcount
        n2 = self.db.conn.execute("DELETE FROM events WHERE content LIKE ?", (f"%{topic}%",)).rowcount
        self.db.conn.commit()
        return n1 + n2