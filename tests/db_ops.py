# db_ops.py
from .types import ConnLike

def write_policy_assignment(conn: ConnLike, profile_id: int, policy: str) -> None:
    conn.execute("INSERT OR REPLACE INTO policy(profile_id, policy) VALUES (?,?)",
                 (profile_id, policy))
    conn.commit()

def read_policy_assignment(conn: ConnLike, profile_id: int) -> str | None:
    row = conn.execute("SELECT policy FROM policy WHERE profile_id=?", (profile_id,)).fetchone()
    return row[0] if row else None