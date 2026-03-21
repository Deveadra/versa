from base.database.sqlite import SQLiteConn


def _table_exists(conn, name: str) -> bool:
    cur = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1", (name,))
    return cur.fetchone() is not None


def test_self_improve_tables_exist(tmp_path):
    db_path = tmp_path / "self_improve_test.db"

    # SQLiteConn should initialize DB + apply migrations on startup (per your logs).
    db = SQLiteConn(str(db_path))
    conn = db.conn

    expected = {
        "dream_runs",
        "dream_artifacts",
        "dream_proposals",
        "capability_gaps",
        "scoreboard",
    }

    missing = [t for t in sorted(expected) if not _table_exists(conn, t)]
    assert not missing, f"Missing tables after migrations: {missing}"
