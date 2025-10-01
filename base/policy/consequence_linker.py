
import json
import random
import re
from datetime import datetime
from base.policy.tone_memory import update_tone_memory


# Hardcoded bootstrap map — acts as seed knowledge
CONSEQUENCE_MAP = {
    "headache": "hydration",
    "fatigue": "sleep",
    "tired": "sleep",
    "leg": "movement",
    "back": "movement",
    "stress": "workload",
    "late": "time_management",
}


def detect_consequence(conn, user_text: str):
    """
    Detect a consequence (user complaint/symptom) and link it to a topic.
    Always return (topic_id, keyword_or_cluster, confidence).
    Dynamically learns new mappings if none exist.
    """
    text = user_text.lower()
    cur = conn.cursor()

    # 1. Hardcoded quick map (bootstrap knowledge)
    for word, topic in CONSEQUENCE_MAP.items():
        if re.search(rf"\b{word}\b", text):
            return topic, word, 0.8

    # 2. DB keyword map
    rows = cur.execute("SELECT keyword, topic_id, confidence FROM consequence_map").fetchall()
    for r in rows:
        if r["keyword"] in text:
            # Increment confidence slightly when used
            new_conf = min(1.0, (r["confidence"] or 0.8) + 0.05)
            cur.execute(
                "UPDATE consequence_map SET confidence=?, last_updated=datetime('now') WHERE keyword=?",
                (new_conf, r["keyword"]),
            )
            conn.commit()
            return r["topic_id"], r["keyword"], new_conf

    # 3. Cluster match
    clusters = cur.execute("SELECT cluster, topic_id, examples FROM complaint_clusters").fetchall()
    for c in clusters:
        examples = json.loads(c["examples"])
        if any(e in text for e in examples):
            # Expand examples with new text if novel
            if user_text not in examples:
                examples.append(user_text)
                cur.execute(
                    """
                    UPDATE complaint_clusters 
                    SET examples=?, last_example=?, last_updated=datetime('now')
                    WHERE cluster=? AND topic_id=?
                    """,
                    (json.dumps(examples), user_text, c["cluster"], c["topic_id"]),
                )
                conn.commit()
            return c["topic_id"], c["cluster"], 0.7

    # 4. Nothing matched → create a *new dynamic mapping* (self-learning)
    inferred_topic = "general"  # fallback topic if not inferred
    cur.execute(
        """
        INSERT INTO consequence_map (keyword, topic_id, confidence, last_updated)
        VALUES (?, ?, ?, datetime('now'))
        """,
        (user_text, inferred_topic, 0.5),
    )
    conn.commit()

    return inferred_topic, user_text, 0.5


def link_consequence(conn, user_text: str):
    """
    Insert complaint, try to link it to ignored advice, update tone memory,
    and improve consequence mappings over time.
    """
    topic, keyword, confidence = detect_consequence(conn, user_text)
    if not topic:
        return False

    cur = conn.cursor()
    cur.execute(
        "INSERT INTO feedback_events (usage_id, kind, note) VALUES (?, ?, ?)",
        (None, "complaint", user_text),
    )
    conn.commit()

    # Try again to confirm consequence
    topic, keyword, confidence = detect_consequence(conn, user_text)
    if not topic:
        return False

    # Find a recent ignored rule for this topic (last 24h)
    row = cur.execute("""
        SELECT rule_id FROM rule_history
        WHERE topic_id=? AND outcome='ignored'
          AND timestamp > datetime('now','-1 day')
        ORDER BY timestamp DESC
        LIMIT 1
    """, (topic,)).fetchone()

    if row:
        update_tone_memory(
            conn,
            topic_id=topic,
            tone="genuine",   # TODO: retrieve actual tone from tone_memory
            outcome="ignored",
            consequence=f"user reported {keyword}"
        )
        return True

    # If no ignored rule, still track it in complaint_clusters
    cur.execute("""
        INSERT INTO complaint_clusters (cluster, topic_id, examples, last_updated, last_example)
        VALUES (?, ?, ?, datetime('now'), ?)
        ON CONFLICT(cluster, topic_id) DO UPDATE SET
            examples=?,
            last_example=?,
            last_updated=datetime('now')
    """, (
        keyword, topic, json.dumps([user_text]), user_text,
        json.dumps([user_text]), user_text
    ))
    conn.commit()

    return False


def style_complaint(complaint: str, mood: str) -> str:
    """Choose how to surface the complaint back to user depending on tone/mood."""
    if not complaint:
        return ""

    if mood in ("sarcastic", "frustrated"):
        return complaint.strip()

    if mood in ("genuine", "patient"):
        return shorten_complaint(complaint)

    if mood in ("smug", "proving"):
        return complaint.strip() if random.random() < 0.7 else shorten_complaint(complaint)

    return shorten_complaint(complaint)


def shorten_complaint(complaint: str) -> str:
    """Simplify complaint to essence (remove filler, shorten length)."""
    styled = complaint.lower()
    styled = re.sub(r"\b(ugh|uh|um|why does|why do|so much|really)\b", "", styled)
    styled = styled.strip().capitalize()

    words = styled.split()
    if len(words) > 8:
        styled = " ".join(words[-5:])

    return styled
