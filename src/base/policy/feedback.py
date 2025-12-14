import sqlite3
import threading
import time


def record_feedback(
    conn: sqlite3.Connection, rule_id: int, topic_id: str, tone: str, context: str, outcome: str
):
    """
    outcome ∈ {"acted","thanks","ignore","angry"}
    """
    conn.execute(
        """
        INSERT INTO rule_history(rule_id, topic_id, tone, context, outcome)
        VALUES(?,?,?,?,?)
    """,
        (rule_id, topic_id, tone, context, outcome),
    )

    # Update EMAs in rule_stats
    row = conn.execute(
        "SELECT ema_success, ema_negative FROM rule_stats WHERE rule_id=?", (rule_id,)
    ).fetchone()
    if not row:
        conn.execute(
            """
            INSERT INTO rule_stats(rule_id, last_fired, fires_today, ema_success, ema_negative)
            VALUES(?, datetime('now'), 0, ?, ?)
        """,
            (
                rule_id,
                0.5 if outcome in ("acted", "thanks") else 0.0,
                0.5 if outcome in ("ignore", "angry") else 0.0,
            ),
        )
    else:
        suc = row["ema_success"] or 0.5
        neg = row["ema_negative"] or 0.5
        alpha = 0.2  # learning rate
        if outcome in ("acted", "thanks"):
            suc = (1 - alpha) * suc + alpha * 1.0
            neg = (1 - alpha) * neg + alpha * 0.0
        elif outcome in ("ignore", "angry"):
            suc = (1 - alpha) * suc + alpha * 0.0
            neg = (1 - alpha) * neg + alpha * 1.0
        conn.execute(
            "UPDATE rule_stats SET ema_success=?, ema_negative=? WHERE rule_id=?",
            (suc, neg, rule_id),
        )

    conn.commit()


def schedule_signal_check(
    conn, rule_id, topic_id, tone, context, signal_names, expect_change, delay=300
):
    """
    After `delay` seconds, check all `signal_names` and call expect_change(values).
    If True, record 'acted'. Otherwise, 'ignore'.
    """

    def check():
        time.sleep(delay)
        vals = {}
        for name in signal_names:
            row = conn.execute("SELECT value FROM context_signals WHERE name=?", (name,)).fetchone()
            if row:
                try:
                    vals[name] = float(row["value"])
                except:
                    vals[name] = row["value"]
        if not vals:
            return
        outcome = "acted" if expect_change(vals) else "ignore"
        record_feedback(conn, rule_id, topic_id, tone, context, outcome)

    threading.Thread(target=check, daemon=True).start()
