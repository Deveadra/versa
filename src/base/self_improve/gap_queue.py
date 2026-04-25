from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

GapType = Literal[
    "unhandled_tool", "tool_error", "missing_capability", "test_failure", "lint_failure"
]
GapStatus = Literal["open", "in_progress", "blocked", "resolved"]


@dataclass
class Gap:
    id: int | None
    gap_type: GapType
    title: str
    detail_json: dict[str, Any]
    created_at: str
    status: GapStatus
    severity: int  # 1-10
    occurrences: int


class GapQueue:
    def __init__(self, conn) -> None:
        self.conn = conn

    def record(
        self, gap_type: GapType, title: str, detail_json: dict[str, Any], severity: int = 5
    ) -> int:
        now = datetime.now(UTC).isoformat()
        cur = self.conn.execute(
            """
            INSERT INTO capability_gaps(gap_type, title, detail_json, created_at, status, severity, occurrences)
            VALUES(?, ?, ?, ?, 'open', ?, 1)
            """,
            (gap_type, title, json.dumps(detail_json), now, int(severity)),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def next_gap(self) -> Gap | None:
        row = self.conn.execute(
            """
            SELECT * FROM capability_gaps
            WHERE status='open'
            ORDER BY severity DESC, occurrences DESC, created_at ASC
            LIMIT 1
            """
        ).fetchone()
        return Gap(**dict(row)) if row else None

    def mark(self, gap_id: int, status: GapStatus) -> None:
        if status not in ("open", "in_progress", "blocked", "resolved"):
            raise ValueError(f"Invalid status: {status}")
        self.conn.execute("UPDATE capability_gaps SET status=? WHERE id=?", (status, gap_id))
        self.conn.commit()
