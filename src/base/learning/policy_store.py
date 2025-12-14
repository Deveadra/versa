from __future__ import annotations

from base.database.sqlite import SQLiteConn


def write_policy_assignment(conn: SQLiteConn, usage_id: int, policy_id: str):
    """Persist mapping usage_id -> policy_id."""
    c = conn.cursor()
    c.execute(
        "INSERT OR REPLACE INTO policy_assignments (usage_id, policy_id) VALUES (?, ?)",
        (usage_id, policy_id),
    )
    conn.conn.commit()


def read_policy_assignment(conn: SQLiteConn, usage_id: int) -> str | None:
    """Return policy_id previously stored for this usage_id, or None."""
    c = conn.cursor()
    c.execute("SELECT policy_id FROM policy_assignments WHERE usage_id = ?", (usage_id,))
    r = c.fetchone()
    return r[0] if r else None
