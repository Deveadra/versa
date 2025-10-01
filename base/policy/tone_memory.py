import sqlite3
from datetime import datetime
from typing import Optional
from base.database.sqlite import SQLiteConn


def get_tone(conn: sqlite3.Connection, topic_id: str, base_tone: str) -> str:
    """
    Decide which tone to use, factoring in last tone + outcome.
    """
    row = conn.execute("SELECT last_tone, last_outcome FROM tone_memory WHERE topic_id=?", (topic_id,)).fetchone()
    if not row:
        return base_tone  # no history → stick with policy decision

    last_tone, last_outcome = row["last_tone"], row["last_outcome"]

    # Simple adaptive logic
    if last_outcome in ("ignore", "angry"):
        if last_tone == "gentle":
            return "persistent"
        elif last_tone == "persistent":
            return "firm"
        else:
            return "firm"  # stay firm if already at firm
    elif last_outcome in ("acted", "thanks"):
        if last_tone == "firm":
            return "persistent"
        elif last_tone == "persistent":
            return "gentle"
        else:
            return "gentle"  # stay gentle if already gentle

    return base_tone

def update_tone_memory(conn: SQLiteConn, topic_id: str, tone: str, outcome: str, consequence: Optional[str] = None):
    cur = conn.cursor()
    row = cur.execute("SELECT id, ignored_count, acted_count FROM tone_memory WHERE topic_id=? AND tone=?",
                      (topic_id, tone)).fetchone()

    if row:
        ignored = row["ignored_count"] + (1 if outcome == "ignored" else 0)
        acted = row["acted_count"] + (1 if outcome == "acted" else 0)
        cur.execute("""
            UPDATE tone_memory
            SET ignored_count=?, acted_count=?, consequence_note=?, last_updated=?
            WHERE id=?
        """, (ignored, acted, consequence or row.get("consequence_note"), datetime.utcnow().isoformat(), row["id"]))
    else:
        cur.execute("""
            INSERT INTO tone_memory (topic_id, tone, ignored_count, acted_count, consequence_note, last_updated)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            topic_id,
            tone,
            1 if outcome == "ignored" else 0,
            1 if outcome == "acted" else 0,
            consequence,
            datetime.utcnow().isoformat()
        ))
    conn.commit()


def choose_tone_for_topic(conn: SQLiteConn, topic_id: str) -> str:
    """
    Pick next tone strategy based on memory:
    - If ignored often + consequence logged → escalate (sarcasm).
    - If acted often → stay genuine.
    """
    cur = conn.cursor()
    rows = cur.execute("SELECT tone, ignored_count, acted_count, consequence_note FROM tone_memory WHERE topic_id=?",
                       (topic_id,)).fetchall()
    if not rows:
        return "genuine"

    # Basic heuristic
    for r in rows:
        if r["ignored_count"] >= 3 and r["consequence_note"]:
            return "sarcastic"
    return "genuine"