
from __future__ import annotations


def test_rollback_discards_uncommitted_write(db) -> None:
    db.conn.execute("CREATE TABLE IF NOT EXISTS tx_test (id INTEGER)")
    db.conn.commit()

    try:
        db.conn.execute("BEGIN")
        db.conn.execute("INSERT INTO tx_test(id) VALUES (1)")
        raise RuntimeError("force rollback")
    except RuntimeError:
        db.conn.rollback()

    rows = db.conn.execute("SELECT COUNT(*) FROM tx_test").fetchone()[0]
    assert rows == 0
