import sqlite3
import os
import datetime

DB_FILE = os.getenv("ULTRON_MEMORY_DB", "ultron_memory.db")

# =====================
# DB INIT
# =====================
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    # Facts table: persistent knowledge (updated if changed)
    c.execute("""
    CREATE TABLE IF NOT EXISTS facts (
        key TEXT PRIMARY KEY,
        value TEXT,
        last_updated TIMESTAMP
    )
    """)

    # History table: rolling 1 month of dialogue/events
    c.execute("""
    CREATE TABLE IF NOT EXISTS history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        text TEXT,
        ts TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()

# =====================
# FACTS
# =====================
def remember_fact(key, value):
    """Insert or update a fact about the user."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
    INSERT INTO facts (key, value, last_updated)
    VALUES (?, ?, ?)
    ON CONFLICT(key) DO UPDATE SET value=excluded.value, last_updated=excluded.last_updated
    """, (key, value, datetime.datetime.utcnow()))
    conn.commit()
    conn.close()

def recall_fact(key):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT value FROM facts WHERE key=?", (key,))
    row = c.fetchone()
    conn.close()
    return row[0] if row else None

def forget_fact(key):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("DELETE FROM facts WHERE key=?", (key,))
    conn.commit()
    conn.close()

def list_facts():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT key, value FROM facts")
    rows = c.fetchall()
    conn.close()
    return {k: v for k, v in rows}

# =====================
# HISTORY
# =====================
def add_history(text):
    """Add an entry to history with timestamp, prune >30 days old."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    now = datetime.datetime.utcnow()
    c.execute("INSERT INTO history (text, ts) VALUES (?, ?)", (text, now))

    # Prune older than 30 days
    cutoff = now - datetime.timedelta(days=30)
    c.execute("DELETE FROM history WHERE ts < ?", (cutoff,))
    conn.commit()
    conn.close()

def get_history(limit=50):
    """Return the most recent N history entries."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT text, ts FROM history ORDER BY ts DESC LIMIT ?", (limit,))
    rows = c.fetchall()
    conn.close()
    return [{"text": t, "ts": ts} for t, ts in rows]

def clear_history():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("DELETE FROM history")
    conn.commit()
    conn.close()
