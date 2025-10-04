# base/learning/feedback.py
from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections.abc import Callable, Iterable
from typing import Any

from base.database.sqlite import SQLiteConn
from base.memory.store import MemoryStore
from base.policy.tone_memory import update_tone_memory
from config.config import settings


def _unwrap_conn(db: SQLiteConn | sqlite3.Connection) -> SQLiteConn:
    """
    Always normalize to SQLiteConn.
    """
    if isinstance(db, SQLiteConn):
        return db
    if isinstance(db, sqlite3.Connection):
        return SQLiteConn(settings.db_path)  # rewrap
    raise TypeError("Unsupported DB connection type passed to feedback module.")


class Feedback:
    """
    Feedback-events writer.
    Normalized to always use a sqlite3.Connection under the hood.
    """

    def __init__(self, conn: SQLiteConn | sqlite3.Connection):
        self.conn = _unwrap_conn(conn)
        # Secondary store (optional) for cross-module integration
        # self.db = SQLiteConn(settings.db_path)
        self.db: SQLiteConn = _unwrap_conn(conn)

    def record(self, usage_id: int, kind: str, note: str | None = None) -> None:
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO feedback_events (usage_id, kind, note) VALUES (?, ?, ?)",
            (usage_id, kind, note),
        )
        self.conn.commit()


# ---------- Rule-specific feedback (engagement loop) ----------


def record_rule_feedback(
    conn: SQLiteConn | sqlite3.Connection,
    rule_id: int,
    topic_id: str,
    tone: str,
    context: str,
    outcome: str,  # "acted" | "thanks" | "ignore" | "angry"
    alpha: float = 0.2,  # EMA learning rate
) -> None:
    """
    Log an outcome for a fired rule, update EMAs in rule_stats,
    and update tone memory for this topic.
    """
    db = _unwrap_conn(conn)
    cur = db.cursor()

    summary = f"User {outcome} {topic_id} reminder. Context: {context}"
    importance = 0.3 if outcome in ("ignore", "angry") else 0.6

    # Memory integration: every feedback is an event Ultron can learn from
    mem = MemoryStore(db)
    mem.add_event(
        summary,
        importance=importance,
        type_="habit_feedback",
        # topic=topic_id,
    )

    # 1) History log
    cur.execute(
        "INSERT INTO rule_history (rule_id, topic_id, tone, context, outcome) VALUES (?,?,?,?,?)",
        (rule_id, topic_id, tone, context, outcome),
    )

    # 2) Update EMAs
    row = cur.execute(
        "SELECT ema_success, ema_negative FROM rule_stats WHERE rule_id=?",
        (rule_id,),
    ).fetchone()

    def _ema_update(suc: float, neg: float, out: str) -> tuple[float, float]:
        if out in ("acted", "thanks"):
            suc = (1 - alpha) * suc + alpha * 1.0
            neg = (1 - alpha) * neg + alpha * 0.0
        elif out in ("ignore", "angry"):
            suc = (1 - alpha) * suc + alpha * 0.0
            neg = (1 - alpha) * neg + alpha * 1.0
        return suc, neg

    if row is None:
        # Seed EMAs depending on outcome
        suc0 = 0.7 if outcome in ("acted", "thanks") else 0.3
        neg0 = 0.3 if outcome in ("acted", "thanks") else 0.7
        cur.execute(
            """
            INSERT INTO rule_stats (rule_id, last_fired, fires_today, ema_success, ema_negative)
            VALUES (?, datetime('now'), 0, ?, ?)
            """,
            (rule_id, suc0, neg0),
        )
    else:
        suc = row["ema_success"] if row["ema_success"] is not None else 0.5
        neg = row["ema_negative"] if row["ema_negative"] is not None else 0.5
        suc, neg = _ema_update(suc, neg, outcome)
        cur.execute(
            "UPDATE rule_stats SET ema_success=?, ema_negative=? WHERE rule_id=?",
            (suc, neg, rule_id),
        )

    db.commit()

    # 3) Tone memory update (per-topic)
    update_tone_memory(db, topic_id, tone, outcome)

    # 4) Reset signals if outcome = acted
    if outcome == "acted":
        rule_row = cur.execute(
            "SELECT reset_signals FROM engagement_rules WHERE id=?",
            (rule_id,),
        ).fetchone()
        if rule_row and rule_row["reset_signals"]:
            try:
                resets = json.loads(rule_row["reset_signals"])
                from base.policy.context_signals import ContextSignals

                ctx_mgr = ContextSignals(SQLiteConn(settings.db_path))
                for sig in resets:
                    ctx_mgr.reset(sig, 0)
            except Exception:
                pass


def schedule_signal_check(
    conn: SQLiteConn | sqlite3.Connection,
    rule_id: int,
    topic_id: str,
    tone: str,
    context: str,
    signal_names: Iterable[str],
    expect_change: Callable[[dict[str, Any]], bool],
    delay: int = 300,
) -> None:
    """
    After `delay` seconds, re-check signals and record feedback automatically.
    """

    db = _unwrap_conn(conn)

    def _fetch_values(names: Iterable[str]) -> dict[str, Any]:
        vals: dict[str, Any] = {}
        placeholders = ",".join(["?"] * len(list(names)))
        query = f"SELECT name, value FROM context_signals WHERE name IN ({placeholders})"
        cur = db.cursor()
        cur.execute(query, list(names))
        for n, v in cur.fetchall():
            try:
                vals[n] = float(v)
            except Exception:
                lv = str(v).strip().lower()
                if lv in ("true", "false"):
                    vals[n] = lv == "true"
                else:
                    vals[n] = v
        return vals

    def _check_later():
        time.sleep(delay)
        try:
            values = _fetch_values(signal_names)
            if not values:
                return
            acted = bool(expect_change(values))
            outcome = "acted" if acted else "ignore"
            record_rule_feedback(db, rule_id, topic_id, tone, context, outcome)
        except Exception:
            # Fail-safe: never crash background threads
            pass

    threading.Thread(target=_check_later, daemon=True).start()
