
import json
from base.policy.topic_manager import get_known_topics
from base.policy.context_signals import ContextSignals
from base.policy.topic_manager import prune_stale_topics



def propose_new_signals_and_rules(conn):
    """
    Ask GPT to suggest new signals and rules based on current history.
    Insert them directly into DB. Logged in audit_log.
    """
    # Fetch context for LLM
    cur = conn.cursor()
    stats = cur.execute("""
        SELECT topic_id, COUNT(*) as fires, 
               SUM(CASE WHEN outcome='acted' THEN 1 ELSE 0 END) as acted,
               SUM(CASE WHEN outcome='ignore' THEN 1 ELSE 0 END) as ignored
        FROM rule_history
        GROUP BY topic_id
        ORDER BY fires DESC
        LIMIT 20
    """).fetchall()

    topics = [r["topic_id"] for r in stats]
    signals = [r["name"] for r in conn.execute("SELECT name FROM context_signals").fetchall()]

    prompt = f"""
    You are Ultron's reasoning module.
    Current signals: {signals}
    Current topics: {topics}
    Engagement history summary: {stats}

    Task: Suggest 1–2 new measurable context signals and rules
    that would help the user. Each signal should be numeric or boolean,
    and each rule should include a condition_json referencing that signal.

    Respond ONLY in strict JSON:
    {{
      "signals": [{{"name": "...", "description": "...", "init_value": 0}}],
      "rules": [{{"name": "...", "topic_id": "...", "condition_json": "...", "tone_strategy_json": "...", "priority": 50}}]
    }}
    """

    reply = ask_brain(prompt)  # wrapper to GPT-5
    try:
        obj = json.loads(reply)
    except Exception:
        return

    # Insert signals
    if "signals" in obj:
        for sig in obj["signals"]:
            conn.execute("""
                INSERT OR IGNORE INTO context_signals (name, value, last_updated)
                VALUES (?, ?, datetime('now'))
            """, (sig["name"], sig.get("init_value", 0)))
            conn.execute("""
                INSERT OR IGNORE INTO audit_log (created_at, rationale)
                VALUES (datetime('now'), ?)
            """, (f"Created new signal '{sig['name']}': {sig['description']}"))

    # Insert rules
    if "rules" in obj:
        for r in obj["rules"]:
            conn.execute("""
                INSERT OR IGNORE INTO engagement_rules
                (name, topic_id, condition_json, tone_strategy_json, priority, enabled)
                VALUES (?,?,?,?,?,1)
            """, (r["name"], r["topic_id"], r["condition_json"], r["tone_strategy_json"], r.get("priority", 50)))
            conn.execute("""
                INSERT OR IGNORE INTO topics (id) VALUES (?)
            """, (r["topic_id"],))
            conn.execute("""
                INSERT OR IGNORE INTO audit_log (created_at, rationale)
                VALUES (datetime('now'), ?)
            """, (f"Created new rule '{r['name']}' for topic '{r['topic_id']}'"))

    conn.commit()
    

def run_dream_cycle(conn):
    # ... existing logic for refining rules, stats, etc.

    # === Prune old/stale topics ===
    pruned = prune_stale_topics(conn, stale_days=90, min_rules=0, min_memories=1)
    
    propose_new_signals_and_rules(conn)
    
    if pruned:
        cur = conn.cursor()
        for topic in pruned:
            cur.execute(
                "INSERT INTO audit_log (created_at, rationale) VALUES (datetime('now'), ?)",
                (f"Pruned stale topic '{topic}' (inactive >90d)",)
            )
        conn.commit()
        
def expand_consequence_map(conn):
    cur = conn.cursor()
    # gather ignored topics + complaints
    complaints = cur.execute("""
        SELECT DISTINCT note FROM feedback_events
        WHERE kind='complaint'
          AND created_at > datetime('now','-7 day')
    """).fetchall()
    topics = cur.execute("""
        SELECT DISTINCT topic_id FROM rule_history
        WHERE outcome='ignored'
          AND timestamp > datetime('now','-7 day')
    """).fetchall()

    prompt = f"""
    Current consequence map: {cur.execute("SELECT keyword, topic_id FROM consequence_map").fetchall()}
    Recent ignored topics: {topics}
    Recent user complaints: {complaints}

    Task: Suggest new or improved keyword→topic mappings that explain user consequences from ignored advice.
    Respond in JSON array: [{{"keyword":"...","topic_id":"...","confidence":0.9}}]
    """

    reply = ask_brain(prompt)
    try:
        new_map = json.loads(reply)
        for m in new_map:
            cur.execute("""
                INSERT OR REPLACE INTO consequence_map (keyword, topic_id, confidence, last_updated)
                VALUES (?,?,?,datetime('now'))
            """, (m["keyword"], m["topic_id"], m.get("confidence", 0.8)))
        conn.commit()
    except Exception:
        pass

def cluster_complaints(conn):
    cur = conn.cursor()
    complaints = cur.execute("""
        SELECT note FROM feedback_events
        WHERE kind='complaint'
          AND created_at > datetime('now','-14 day')
    """).fetchall()

    if not complaints:
        return

    complaint_texts = [c["note"] for c in complaints]
    prompt = f"""
    Cluster these user complaints into categories and map to known topics.
    Known topics: {cur.execute("SELECT DISTINCT topic_id FROM engagement_rules").fetchall()}
    Complaints: {complaint_texts}

    Respond as JSON array:
    [{{"cluster":"aches","examples":["back hurts","headache"],"topic_id":"movement"}}]
    """
    cur.execute("""
        INSERT INTO complaint_clusters (cluster, topic_id, examples, last_updated, last_example)
        VALUES (?,?,?,?,?)
    """, (
        c["cluster"],
        c["topic_id"],
        json.dumps(c["examples"]),
        datetime.utcnow().isoformat(),
        c["examples"][-1] if c["examples"] else None
    ))


    reply = ask_brain(prompt)
    try:
        clusters = json.loads(reply)
        for c in clusters:
            cur.execute("""
                INSERT INTO complaint_clusters (cluster, topic_id, examples, last_updated)
                VALUES (?,?,?,datetime('now'))
            """, (c["cluster"], c["topic_id"], json.dumps(c["examples"])))
        conn.commit()
    except Exception:
        pass
