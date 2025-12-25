from __future__ import annotations


def test_events_fts_can_search_after_rebuild(db) -> None:
    db.conn.execute(
        "INSERT INTO events(content, ts, importance, type) VALUES(?, ?, ?, ?)",
        ("hello ultron world", "2025-01-01T00:00:00", 0.5, "event"),
    )
    db.conn.commit()

    # External content FTS often needs rebuild without triggers
    db.conn.execute("INSERT INTO events_fts(events_fts) VALUES('rebuild');")

    rows = db.conn.execute(
        "SELECT rowid, content FROM events_fts WHERE events_fts MATCH ?;",
        ("ultron",),
    ).fetchall()

    assert len(rows) >= 1
