
import sqlite3
from typing import List, Union
from datetime import timedelta, datetime

from base.database.sqlite import SQLiteConn


def get_known_topics(conn: Union[SQLiteConn, sqlite3.Connection]) -> List[str]:
    """
    Return all topics Ultron currently knows about.
    Populated automatically by dream cycle inserts into topics table.
    """
    cur = conn.cursor()
    rows = cur.execute("SELECT id FROM topics ORDER BY created_at ASC").fetchall()
    return [r["id"] for r in rows]

def prune_stale_topics(
    conn: Union[SQLiteConn, sqlite3.Connection],
    stale_days: int = 90,
    min_rules: int = 0,
    min_memories: int = 0
) -> List[str]:
    """
    Remove topics that have not been referenced recently in rules or memories.

    Args:
        stale_days: how many days of inactivity before a topic is prunable
        min_rules: minimum number of active rules that must reference the topic to keep it
        min_memories: minimum number of memory events tagged with this topic to keep it

    Returns:
        List of topic IDs that were pruned
    """
    cur = conn.cursor()

    cutoff = (datetime.utcnow() - timedelta(days=stale_days)).isoformat()

    # Find candidate topics
    rows = cur.execute("SELECT id, created_at FROM topics").fetchall()
    pruned = []

    for r in rows:
        topic = r["id"]

        # Check last memory use
        mem = cur.execute(
            "SELECT MAX(created_at) as last_used FROM memory_events WHERE topic=?",
            (topic,),
        ).fetchone()
        last_mem = mem["last_used"] if mem and mem["last_used"] else None

        # Check active rules
        rules = cur.execute(
            "SELECT COUNT(*) as cnt FROM engagement_rules WHERE topic_id=? AND enabled=1",
            (topic,),
        ).fetchone()
        rule_count = rules["cnt"] if rules else 0

        # Check memory count
        mem_count_row = cur.execute(
            "SELECT COUNT(*) as cnt FROM memory_events WHERE topic=?",
            (topic,),
        ).fetchone()
        mem_count = mem_count_row["cnt"] if mem_count_row else 0

        # Decide if stale
        if rule_count <= min_rules and mem_count <= min_memories:
            if not last_mem or last_mem < cutoff:
                cur.execute("DELETE FROM topics WHERE id=?", (topic,))
                pruned.append(topic)

    conn.commit()
    return pruned