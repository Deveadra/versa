
import random
import re
from datetime import datetime, timedelta
from base.policy.tone_memory import update_tone_memory

# crude keyword mapping of consequence → likely topic
CONSEQUENCE_MAP = {
    "headache": "hydration",
    "fatigue": "sleep",
    "tired": "sleep",
    "leg": "movement",
    "back": "movement",
    "stress": "workload",
    "late": "time_management"
}

def detect_consequence(conn, user_text: str):
    """Look up consequence keywords from DB instead of hardcoded map."""
    text = user_text.lower()
    cur = conn.cursor()
    rows = cur.execute("SELECT keyword, topic_id, confidence FROM consequence_map").fetchall()
    text = user_text.lower()
    for word, topic in CONSEQUENCE_MAP.items():
        if re.search(rf"\b{word}\b", text):
            return topic, word
    # for r in rows:
    #     if re.search(rf"\b{re.escape(r['keyword'])}\b", text):
    #         return r["topic_id"], r["keyword"], r["confidence"]
    # 1. Direct keyword map
    for r in rows:
        if r["keyword"] in text:
            return r["topic_id"], r["keyword"], 0.9
    # 2. Cluster match
    clusters = cur.execute("SELECT cluster, topic_id, examples FROM complaint_clusters").fetchall()
    for c in clusters:
        examples = json.loads(c["examples"])
        if any(e in text for e in examples):
            return c["topic_id"], c["cluster"], 0.7
    return None, None, None


def link_consequence(conn, user_text: str):
    topic, word, confidence = detect_consequence(conn, user_text)
    if not topic:
        return False

    cur = conn.cursor()
    cur.execute(
        "INSERT INTO feedback_events (usage_id, kind, note) VALUES (?, ?, ?)",
        (None, "complaint", user_text)
    )
    conn.commit()

    # 2) Try to link to ignored advice
    topic, word, confidence = detect_consequence(conn, user_text)
    if not topic:
        return False
      
    # find a recent ignored rule for this topic (last 24h)
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
            tone="genuine",   # whatever tone was last used
            outcome="ignored",
            consequence=f"user reported {word}"
        )
        return True
    cur.execute("""
        UPDATE complaint_clusters
        SET last_example=?, last_updated=datetime('now')
        WHERE topic_id=? AND cluster=?
    """, (user_text, topic, cluster))
    conn.commit()

    return False
  
import random
import re


def style_complaint(complaint: str, mood: str) -> str:
    """
    Decide whether to return complaint verbatim or styled, based on mood.
    - sarcastic/frustrated → verbatim (to rub it in)
    - patient/genuine → styled (softer, more conversational)
    - smug/proving → mostly verbatim, with occasional styled for flair
    """
    if not complaint:
        return ""

    if mood in ("sarcastic", "frustrated"):
        return complaint.strip()

    if mood in ("genuine", "patient"):
        return shorten_complaint(complaint)

    if mood in ("smug", "proving"):
        import random
        return complaint.strip() if random.random() < 0.7 else shorten_complaint(complaint)

    # default fallback
    return shorten_complaint(complaint)


def shorten_complaint(complaint: str) -> str:
    """Simplify complaint to essence (remove filler, shorten length)."""
    import re
    styled = complaint.lower()
    styled = re.sub(r"\b(ugh|uh|um|why does|why do|so much|really)\b", "", styled)
    styled = styled.strip().capitalize()

    words = styled.split()
    if len(words) > 8:
        styled = " ".join(words[-5:])

    return styled
