# base/memory/store.py
from __future__ import annotations

import json
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from base.database.sqlite import SQLiteConn
from base.memory.scoring import assess_importance
from config.config import settings

DB_PATH = Path("memory.db")


class MemoryStore:
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
        self.init_db()
        self._subscribers = []

    # ---------- schema ----------
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

        # Legacy "memories" table for whole exchanges
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                type TEXT,
                content TEXT,
                response TEXT
            )
            """
        )

        # Events (atomic things worth recalling)
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

        # Full-text search index linked to events.content
        try:
            cur.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS events_fts
                USING fts5(content, content='events', content_rowid='id')
                """
            )

            # Keep FTS index in sync automatically
            cur.executescript(
                """
                CREATE TRIGGER IF NOT EXISTS events_ai AFTER INSERT ON events BEGIN
                  INSERT INTO events_fts(rowid, content) VALUES (new.id, new.content);
                END;

                CREATE TRIGGER IF NOT EXISTS events_ad AFTER DELETE ON events BEGIN
                  INSERT INTO events_fts(events_fts, rowid, content) VALUES('delete', old.id, old.content);
                END;

                CREATE TRIGGER IF NOT EXISTS events_au AFTER UPDATE ON events BEGIN
                  INSERT INTO events_fts(events_fts, rowid, content) VALUES('delete', old.id, old.content);
                  INSERT INTO events_fts(rowid, content) VALUES (new.id, new.content);
                END;
                """
            )

            self._fts_enabled = True
        except sqlite3.OperationalError as e:
            from loguru import logger

            logger.error(f"FTS init failed: {e}")
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

    def list_facts(self) -> list[tuple[str, str]]:
        cur = self.conn.execute("SELECT key, value FROM facts ORDER BY key")
        return [(r["key"], r["value"]) for r in cur.fetchall()]

    def forget(self, topic: str) -> int:
        """Delete facts and events containing the given topic string."""
        n1 = (
            self.conn.execute(
                "DELETE FROM facts WHERE key LIKE ? OR value LIKE ?",
                (f"%{topic}%", f"%{topic}%"),
            ).rowcount
            or 0
        )
        n2 = (
            self.conn.execute(
                "DELETE FROM events WHERE content LIKE ?",
                (f"%{topic}%",),
            ).rowcount
            or 0
        )
        self.conn.commit()
        return n1 + n2

    def subscribe(self, callback):
        """
        Register a callback to be called whenever a new event is added.
        Callback signature: fn(content: str, ts: str, type_: str, importance: float).
        """
        self._subscribers.append(callback)

    # ---------- events ----------
    def add_event(self, content: str, importance: float = 0.0, type_: str = "event") -> int:
        ts = datetime.utcnow().isoformat()
        cur = self.conn.execute(
            "INSERT INTO events(content, ts, importance, type) VALUES(?, ?, ?, ?)",
            (content, ts, importance, type_),
        )
        rowid = cur.lastrowid
        if rowid is None:
            raise RuntimeError("Failed to insert event: lastrowid is None")

        if self._fts_enabled:
            try:
                self.conn.execute(
                    "INSERT INTO events_fts(rowid, content) VALUES(?, ?)",
                    (rowid, content),
                )
            except sqlite3.OperationalError:
                self._fts_enabled = False

        self.conn.commit()

        for cb in self._subscribers:
            try:
                cb(content=content, ts=ts, type_=type_, importance=importance)
            except Exception as e:
                from loguru import logger

                logger.error(f"Subscriber failed: {e}")

        return int(rowid)

    def add_diagnostic_event(
        self,
        *,
        mode: str,
        fix: bool,
        base: Optional[str],
        diag_output: str,
        issues: List[Dict[str, Any]],
        benchmarks: List[Dict[str, Any]],
        laggy: bool,
        started_at_iso: str,
        duration_ms: float,
    ) -> None:
        """
        Persist a structured diagnostic record into memory.
        Uses the existing event log mechanism so nothing else in your stack breaks.
        """
        payload = {
            "type": "diagnostic",
            "mode": mode,
            "fix": fix,
            "base": base,
            "tool_output": diag_output[-8000:],  # cap to avoid massive blobs
            "issues": issues,
            "benchmarks": benchmarks,
            "laggy": laggy,
            "started_at": started_at_iso,
            "duration_ms": duration_ms,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "version": 1,
        }
        # Important: keep the same event API your code already uses
        # (content text field + type_ discriminator).
        try:
            self.add_event(
                content=json.dumps(payload, ensure_ascii=False),
                importance=0.0,
                type_="diagnostic",
            )
        except Exception as e:
            # Graceful fallback: don't crash the run; keep an audit trail.
            try:
                with open("diagnostics.log", "a", encoding="utf-8") as f:
                    f.write(json.dumps({"fallback": True, **payload}) + "\n")
            except Exception:
                pass

    def recent_diagnostics(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Return the most recent diagnostic events (decoded).
        Falls back to reading diagnostics.log if the event store is unavailable.
        """
        out: List[Dict[str, Any]] = []
        try:
            # If you have a native list_events(type_=...) use it here.
            # Otherwise this uses a common content/type_ pattern.
            rows = self.list_events(type_="diagnostic", limit=limit)  # <-- your existing helper if present
            for r in rows:
                # r["content"] expected to be a JSON string from add_diagnostic_event
                try:
                    out.append(json.loads(r.get("content", "{}")))
                except Exception:
                    continue
            return out
        except Exception:
            # Fallback to file if DB listing fails
            try:
                with open("diagnostics.log", "r", encoding="utf-8") as f:
                    lines = f.readlines()[-limit:]
                for ln in lines:
                    try:
                        out.append(json.loads(ln))
                    except Exception:
                        continue
                return out
            except Exception:
                return out  # empty if nothing works

    def last_diagnostic(self) -> Optional[Dict[str, Any]]:
        items = self.recent_diagnostics(limit=1)
        return items[0] if items else None

    def diagnostics_since(self, iso_timestamp: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Filter diagnostics since a given ISO timestamp.
        """
        try:
            since = datetime.fromisoformat(iso_timestamp.replace("Z", "+00:00"))
        except Exception:
            return self.recent_diagnostics(limit=limit)

        results: List[Dict[str, Any]] = []
        for ev in self.recent_diagnostics(limit=limit):
            try:
                ts = datetime.fromisoformat(ev.get("created_at", "").replace("Z", "+00:00"))
                if ts >= since:
                    results.append(ev)
            except Exception:
                continue
        return results

    def maybe_store_text(self, text: str, explicit_type: str | None = None) -> bool:
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
    # def keyword_search(self, query, limit):
    #     limit = 5
    #     # Strip commas from the query
    #     query = query.replace(',', '')
    #     # Existing functionality to search using FTS triggers or LIKE fallback
    #     # Add your FTS implementation here
    #     pass
    #     query = query.replace(',', '')  # Strip commas from the query
    #     # Existing functionality for FTS triggers and LIKE fallback goes here
    #     if self._fts_enabled:
    #         try:
    #             sql = f"SELECT content FROM events_fts WHERE events_fts MATCH ? LIMIT {int(limit)}"
    #             cur = self.conn.execute(sql, (query,))
    #             return [r[0] for r in cur.fetchall()]
    #         except sqlite3.OperationalError as e:
    #             from loguru import logger
    #             logger.error(f"FTS search failed, falling back to LIKE: {e}")
    #             self._fts_enabled = False
    #             # fallback to LIKE if FTS errors out
    #     # fallback: LIKE
    #     cur = self.conn.execute(
    #         "SELECT content FROM events WHERE content LIKE ? ORDER BY id DESC LIMIT ?",
    #         (f"%{query}%", limit),
    #     )
    #     # return [dict(r) for r in cur.fetchall()]
    #     return [r[0] for r in cur.fetchall()]

    def keyword_search(self, query: str, limit: int = 5) -> list[dict]:
        """
        Search memory for a keyword or phrase.
        Tries FTS first, falls back to LIKE if needed.
        """
        try:
            cur = self.conn.execute(
                "SELECT * FROM memories WHERE content MATCH ? LIMIT ?", (query, limit)
            )
            return [dict(r) for r in cur.fetchall()]
        except Exception:
            cur = self.conn.execute(
                "SELECT * FROM memories WHERE content LIKE ? LIMIT ?", (f"%{query}%", limit)
            )
            return [dict(r) for r in cur.fetchall()]

    # ---------- Legacy helpers for backward compatibility ----------

    def _connect_for_compat(self) -> sqlite3.Connection:
        return sqlite3.connect(settings.db_path or str(DB_PATH), check_same_thread=False)

    def init_db(self) -> None:
        with self._connect_for_compat() as conn:
            cur = conn.cursor()
            # run the same schema setup here if needed
            # or simply no-op since _ensure_schema already covers this
            conn.commit()

    def save_memory(self, memory: dict) -> None:
        ts = memory.get("timestamp") or datetime.utcnow().isoformat()
        typ = memory.get("type", "event")
        content = memory.get("content", "")
        response = memory.get("response", "")

        with self._connect_for_compat() as conn:
            conn.execute(
                "INSERT INTO memories(timestamp, type, content, response) VALUES(?, ?, ?, ?)",
                (ts, typ, content, response),
            )
            conn.commit()
