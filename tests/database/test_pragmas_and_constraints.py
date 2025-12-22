# tests/database/test_pragmas_and_constraints.py

from __future__ import annotations

import sqlite3
import pytest


def test_journal_mode_is_wal(db) -> None:
    mode = db.conn.execute("PRAGMA journal_mode;").fetchone()[0]
    assert str(mode).lower() == "wal"


def test_foreign_keys_enabled(db) -> None:
    fk = db.conn.execute("PRAGMA foreign_keys;").fetchone()[0]
    assert fk == 1


def test_habits_key_is_unique(db) -> None:
    db.conn.execute("INSERT INTO habits(key, count, score) VALUES(?, ?, ?)", ("sleep", 1, 0.2))
    db.conn.commit()
    with pytest.raises(sqlite3.IntegrityError):
        db.conn.execute("INSERT INTO habits(key, count, score) VALUES(?, ?, ?)", ("sleep", 1, 0.2))


def test_topics_policy_check_constraint(db) -> None:
    db.conn.execute("INSERT INTO topics(topic_id, policy) VALUES(?, ?)", ("hydration", "adaptive"))
    db.conn.commit()
    with pytest.raises(sqlite3.IntegrityError):
        db.conn.execute("INSERT INTO topics(topic_id, policy) VALUES(?, ?)", ("bad", "invalid_policy"))


def test_context_signals_type_check_constraint(db) -> None:
    db.conn.execute(
        "INSERT INTO context_signals(name, type) VALUES(?, ?)",
        ("stress_level", "integer"),
    )
    db.conn.commit()
    with pytest.raises(sqlite3.IntegrityError):
        db.conn.execute(
            "INSERT INTO context_signals(name, type) VALUES(?, ?)",
            ("nope", "string"),
        )
