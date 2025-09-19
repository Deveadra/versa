# base/memory/recall.py
import sqlite3
from pathlib import Path

DB_PATH = Path("memory.db")

def recall_relevant(query: str, limit=5):
    """
    Return recent memories that may be relevant to the query.
    Simple keyword-based matching for now.
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT timestamp, type, content, response
        FROM memories
        ORDER BY id DESC
        LIMIT 200
    """)
    rows = cur.fetchall()
    conn.close()

    # Naive matching: return ones containing overlapping words
    query_tokens = set(query.lower().split())
    matches = []
    for ts, mtype, content, response in rows:
        text_blob = f"{content} {response}".lower()
        if any(tok in text_blob for tok in query_tokens):
            matches.append((ts, mtype, content, response))
            if len(matches) >= limit:
                break

    return matches

def format_memories(memories):
    """
    Convert memory tuples into a summary string that can be injected into the conversation.
    """
    if not memories:
        return None
    lines = []
    for ts, mtype, content, response in memories:
        lines.append(f"[{ts}] You said: '{content}' | Ultron replied: '{response}'")
    return "\n".join(lines)
