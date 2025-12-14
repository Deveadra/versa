import sqlite3
from datetime import datetime
from typing import Any

from base.database.sqlite import SQLiteConn


def _unwrap_conn(db: SQLiteConn | sqlite3.Connection) -> sqlite3.Connection:
    if isinstance(db, sqlite3.Connection):
        return db
    if hasattr(db, "conn") and isinstance(db.conn, sqlite3.Connection):
        return db.conn
    raise TypeError("Unsupported DB connection type passed to context_signals.")


class ContextSignals:
    """
    Registry and manager for context signals.
    Signals are dynamic: Ultron can invent new ones during dream cycle.
    Each signal has:
      - name (str)
      - value (str/float/bool)
      - last_updated (timestamp)
    Registry and manager for context signals.
    Now supports type-aware signals:
      - counter: increments with time (minutes, hours, etc.)
      - boolean: true/false states
      - derived: values calculated from other signals
    """

    def __init__(self, conn: SQLiteConn | sqlite3.Connection):
        self.conn = _unwrap_conn(conn)

    # ---- CRUD ----

    def upsert(self, name: str, value: Any, type_: str = "counter", description: str = ""):
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO context_signals (name, value, type, description, last_updated)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
              value=excluded.value,
              type=excluded.type,
              description=excluded.description,
              last_updated=excluded.last_updated
            """,
            (name, str(value), type_, description, datetime.utcnow().isoformat()),
        )
        self.conn.commit()

    def get(self, name: str) -> str | None:
        cur = self.conn.cursor()
        row = cur.execute("SELECT value FROM context_signals WHERE name=?", (name,)).fetchone()
        return row["value"] if row else None

    def all(self) -> dict[str, Any]:
        cur = self.conn.cursor()
        rows = cur.execute("SELECT name, value, type FROM context_signals").fetchall()
        return {r["name"]: {"value": r["value"], "type": r["type"]} for r in rows}

    def delete(self, name: str):
        self.conn.execute("DELETE FROM context_signals WHERE name=?", (name,))
        self.conn.commit()

    # ---- Helper Updates ----

    def increment(self, name: str, delta: float = 1.0):
        """Increment numeric signals of type counter."""
        cur = self.conn.cursor()
        row = cur.execute("SELECT value FROM context_signals WHERE name=?", (name,)).fetchone()
        try:
            val = float(row["value"]) if row else 0.0
        except Exception:
            val = 0.0
        self.upsert(name, val + delta, type_="counter")

    def reset(self, name: str, to: float = 0.0):
        self.upsert(name, to)

    def mark_true(self, name: str):
        self.upsert(name, True, type_="boolean")

    def mark_false(self, name: str):
        self.upsert(name, False, type_="boolean")
