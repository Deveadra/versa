# base/memory/recall.py
import json
import sqlite3
from pathlib import Path

from base.llm.brain import ask_brain
from base.policy.topic_manager import get_known_topics

DB_PATH = Path("memory.db")


def detect_topics_with_llm(user_text: str, known_topics: list[str]) -> list[str]:
    """
    Ask GPT which of the known topics are relevant to the given user_text.
    """
    prompt = f"""
    User said: "{user_text}"
    Known topics: {", ".join(known_topics)}
    Which of these topics are relevant to the user's statement? 
    Respond with a JSON list of topic ids (from the known topics) that apply.
    """
    reply = ask_brain(prompt)  # wrapper around GPT call
    try:
        return json.loads(reply)
    except Exception:
        return []


def recall_relevant(query: str, limit=5):
    """
    Return recent memories that may be relevant to the query.
    Simple keyword-based matching for now.
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT timestamp, type, content, response
        FROM memories
        ORDER BY id DESC
        LIMIT 200
    """
    )
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


def recall_habit_feedback(conn, user_text: str, limit: int = 5):
    known = [r["topic_id"] for r in conn.execute("SELECT DISTINCT topic_id FROM engagement_rules")]
    matched = detect_topics_with_llm(user_text, known)
    known_topics = get_known_topics(conn)
    matched = detect_topics_with_llm(user_text, known_topics)

    if not matched:
        return []

    placeholders = ",".join("?" for _ in matched)
    rows = conn.execute(
        f"""
        SELECT content, created_at, topic
        FROM memory_events
        WHERE type='habit_feedback'
          AND topic IN ({placeholders})
        ORDER BY created_at DESC
        LIMIT {limit}
        """,
        matched,
    ).fetchall()

    return [f"[Habit feedback:{r['topic']}] {r['content']} ({r['created_at']})" for r in rows]


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
