# tests/database/test_migrations_tracking.py

from __future__ import annotations

from tests.database.schema_contract import REQUIRED_MIGRATIONS


def test_schema_migrations_table_exists(db) -> None:
    row = db.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_migrations';"
    ).fetchone()
    assert row is not None


def test_migration_recorded(db) -> None:
    rows = db.conn.execute("SELECT filename FROM schema_migrations").fetchall()
    applied = {r["filename"] for r in rows}
    missing = REQUIRED_MIGRATIONS - applied
    assert not missing, f"Missing applied migrations: {sorted(missing)}"


def test_migration_not_reapplied_on_reconnect(tmp_path) -> None:
    from base.database.sqlite import SQLiteConn

    db_path = tmp_path / "reconnect.db"

    c1 = SQLiteConn(str(db_path))
    count1 = c1.conn.execute("SELECT COUNT(*) AS n FROM schema_migrations").fetchone()["n"]
    c1.close()

    c2 = SQLiteConn(str(db_path))
    count2 = c2.conn.execute("SELECT COUNT(*) AS n FROM schema_migrations").fetchone()["n"]
    c2.close()

    assert count2 == count1
