from __future__ import annotations

import json
import sqlite3
from typing import Any

from loguru import logger


class ContextManager:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    # =============== Raw signals ===============

    def set_signal(self, name: str, value: Any, confidence: float = 1.0, source: str = "system"):
        self.conn.execute(
            """
            INSERT INTO context_signals(name, value, confidence, source, last_updated)
            VALUES(?,?,?,?,CURRENT_TIMESTAMP)
            ON CONFLICT(name) DO UPDATE SET
                value=excluded.value,
                confidence=excluded.confidence,
                source=excluded.source,
                last_updated=CURRENT_TIMESTAMP
            """,
            (name, str(value), confidence, source),
        )
        self.conn.commit()

    def get_signal(self, name: str) -> dict | None:
        row = self.conn.execute("SELECT * FROM context_signals WHERE name=?", (name,)).fetchone()
        return dict(row) if row else None

    def all_signals(self) -> dict[str, dict]:
        rows = self.conn.execute("SELECT * FROM context_signals").fetchall()
        return {r["name"]: dict(r) for r in rows}

    # =============== Derived signals ===============

    def define_derived_signal(self, name: str, definition: dict):
        """
        Example definition:
        {"all":[{"signal":"hydration_gap","value":"true"},{"signal":"time_of_day","gte":"21:00"}]}
        """
        self.conn.execute(
            "INSERT INTO derived_signals(name, definition) VALUES(?,?) ON CONFLICT(name) DO UPDATE SET definition=excluded.definition",
            (name, json.dumps(definition)),
        )
        self.conn.commit()

    def eval_derived_signals(self) -> dict[str, bool]:
        """
        Evaluate all derived signals against current signals.
        """
        base = self.all_signals()
        derived = {}
        rows = self.conn.execute("SELECT * FROM derived_signals").fetchall()
        for r in rows:
            try:
                defn = json.loads(r["definition"])
                ok = self._eval_defn(defn, base)
                derived[r["name"]] = ok
            except Exception as e:
                logger.error(f"Bad derived signal {r['name']}: {e}")
        return derived

    def _eval_defn(self, defn: dict, base: dict[str, dict]) -> bool:
        # only "all"/"any" supported for now
        if "all" in defn:
            return all(self._check_condition(c, base) for c in defn["all"])
        if "any" in defn:
            return any(self._check_condition(c, base) for c in defn["any"])
        return False

    def _check_condition(self, cond: dict, base: dict[str, dict]) -> bool:
        sig = cond["signal"]
        if sig not in base:
            return False
        val = base[sig]["value"]
        if "value" in cond:
            return str(val).lower() == str(cond["value"]).lower()
        # future: numeric/time comparisons
        return False

    # =============== Dream cycle helpers ===============

    def prune_stale_signals(self, days: int = 30):
        self.conn.execute(
            "DELETE FROM context_signals WHERE last_updated < datetime('now', ?)",
            (f"-{days} days",),
        )
        self.conn.commit()

    def list_derived(self) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM derived_signals").fetchall()
        return [dict(r) for r in rows]
