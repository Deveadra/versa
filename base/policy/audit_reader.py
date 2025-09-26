import sqlite3
from typing import List, Dict

def recent_audits(conn: sqlite3.Connection, limit: int = 10) -> List[Dict]:
    rows = conn.execute("""
        SELECT rule_name, topic_id, rationale, details_json, created_at
        FROM rule_audit
        ORDER BY created_at DESC
        LIMIT ?
    """, (limit,)).fetchall()
    return [dict(r) for r in rows]
