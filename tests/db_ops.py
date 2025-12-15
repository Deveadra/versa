# tests/db_ops.py
from __future__ import annotations

from typing import Any, Protocol


class ConnLike(Protocol):
    def execute(self, sql: str, params: tuple[Any, ...] = ...) -> Any: ...
    def commit(self) -> None: ...


def write_policy_assignment(conn: ConnLike, profile_id: int, policy: str) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO policy(profile_id, policy) VALUES (?, ?)",
        (profile_id, policy),
    )
    conn.commit()


def read_policy_assignment(conn: ConnLike, profile_id: int) -> str | None:
    row = conn.execute("SELECT policy FROM policy WHERE profile_id=?", (profile_id,)).fetchone()
    return row[0] if row else None
