# tests/database/test_connection.py

from __future__ import annotations

import sqlite3
import pytest


def test_open_close_connection(db) -> None:
    assert isinstance(db.conn, sqlite3.Connection)
    cur = db.conn.execute("SELECT 1")
    assert cur.fetchone()[0] == 1

    db.close()
    with pytest.raises(sqlite3.ProgrammingError):
        db.conn.execute("SELECT 1")  # type: ignore[attr-defined]


def test_invalid_path_connection(tmp_path) -> None:
    # Passing a directory path should fail
    with pytest.raises(sqlite3.OperationalError):
        from base.database.sqlite import SQLiteConn
        SQLiteConn(str(tmp_path))
