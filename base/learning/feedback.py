
from __future__ import annotations
from typing import Optional, Iterable, Callable, Dict, Any, Union
import sqlite3
import threading
import time

# Use the path your project uses elsewhere
from base.database.sqlite import SQLiteConn
from base.policy.tone_memory import update_tone_memory


def _unwrap_conn(db: Union[SQLiteConn, sqlite3.Connection]) -> sqlite3.Connection:
    """
    Accept either your SQLiteConn wrapper or a raw sqlite3.Connection
    and return a sqlite3.Connection.
    """
    if isinstance(db, sqlite3.Connection):
        return db
    if hasattr(db, "conn") and isinstance(db.conn, sqlite3.Connection):
        return db.conn
    raise TypeError("Unsupported DB connection type passed to feedback module.")


class Feedback:
    """
    Generic feedback-events writer (your existing usage logs).
    Left intact, but normalized to always use a sqlite3.Connection under the hood.
    """
    def __init__(self, conn: Union[SQLiteConn, sqlite3.Connection]):
        self.conn = _unwrap_conn(conn)

    def record(self, usage_id: int, kind: str, note: Optional[str] = None):
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO feedback_events (usage_id, kind, note) VALUES (?, ?, ?)",
            (usage_id, kind, note),
        )
        self.conn.commit()


# ---------- Rule-specific feedback (engagement loop) ----------

def record_rule_feedback(
    conn: Union[SQLiteConn, sqlite3.Connection],
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
    mem = MemoryStore(db)
    mem.add_event(
        summary,
        importance=importance,
        type_="habit_feedback",
        topic=topic_id
    )

    # 1) History
    cur.execute(
        "INSERT INTO rule_history (rule_id, topic_id, tone, context, outcome) VALUES (?,?,?,?,?)",
        (rule_id, topic_id, tone, context, outcome),
    )

    # 2) Update EMAs
    row = cur.execute(
        "SELECT ema_success, ema_negative FROM rule_stats WHERE rule_id=?",
        (rule_id,),
    ).fetchone()

    # 3) Tone memory update
    update_tone_memory(db, topic_id, tone, outcome)

    # 4) Reset signals if this rule specifies any
    if outcome == "acted":
        rule_row = cur.execute(
            "SELECT reset_signals FROM engagement_rules WHERE id=?",
            (rule_id,)
        ).fetchone()
        if rule_row and rule_row["reset_signals"]:
            try:
                resets = json.loads(rule_row["reset_signals"])
                for sig in resets:
                    from base.policy.context_signals import ContextSignals
                    ctx_mgr = ContextSignals(db)
                    ctx_mgr.reset(sig, 0)
            except Exception:
                pass
            
    def _ema_update(suc: float, neg: float, out: str) -> tuple[float, float]:
        # positive signals
        if out in ("acted", "thanks"):
            suc = (1 - alpha) * suc + alpha * 1.0
            neg = (1 - alpha) * neg + alpha * 0.0
        # negative signals
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


def schedule_signal_check(
    conn: Union[SQLiteConn, sqlite3.Connection],
    rule_id: int,
    topic_id: str,
    tone: str,
    context: str,
    signal_names: Iterable[str],
    expect_change: Callable[[Dict[str, Any]], bool],
    delay: int = 300,
) -> None:
    """
    After `delay` seconds, read all named signals from context_signals, then
    call expect_change({signal_name: value}). If True → record 'acted', else 'ignore'.

    - signal_names: one or more context signal names the rule condition was based on
    - expect_change: function derived from the rule condition that returns True when
                     the "improvement" is observed (e.g., leaving a risky range).
    """
    db = _unwrap_conn(conn)

    def _fetch_values(names: Iterable[str]) -> Dict[str, Any]:
        vals: Dict[str, Any] = {}
        placeholders = ",".join(["?"] * len(list(names)))
        query = f"SELECT name, value FROM context_signals WHERE name IN ({placeholders})"
        cur = db.cursor()
        cur.execute(query, list(names))
        for n, v in cur.fetchall():
            # coerce to float where possible, keep raw otherwise
            try:
                vals[n] = float(v)
            except Exception:
                # boolean-ish strings -> normalize
                lv = str(v).strip().lower()
                if lv in ("true", "false"):
                    vals[n] = (lv == "true")
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
            # Fail-safe: never crash the process because of background thread errors
            pass

    threading.Thread(target=_check_later, daemon=True).start()
