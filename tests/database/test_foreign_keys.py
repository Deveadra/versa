# tests/database/test_foreign_keys.py

from __future__ import annotations

import sqlite3

import pytest


def test_feedback_events_fk_enforced(db) -> None:
    # usage_log(id) must exist first
    with pytest.raises(sqlite3.IntegrityError):
        db.conn.execute(
            "INSERT INTO feedback_events(usage_id, kind, note) VALUES(?, ?, ?)",
            (9999, "negative", "nope"),
        )
