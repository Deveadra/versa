import pytest


def test_add_event_is_searchable(memstore):
    eid = memstore.add_event("alpha bravo charlie", importance=1.0, type_="event")
    assert isinstance(eid, int)

    hits = memstore.keyword_search("alpha", limit=5)
    assert any("alpha bravo charlie" == h for h in hits)


def test_fts_updates_on_update(memstore):
    eid = memstore.add_event("original text", importance=1.0, type_="event")

    # Update the row; FTS trigger should follow
    memstore.conn.execute("UPDATE events SET content=? WHERE id=?", ("updated text", eid))
    memstore.conn.commit()

    hits_old = memstore.keyword_search("original", limit=5)
    assert all("original text" != h for h in hits_old)

    hits_new = memstore.keyword_search("updated", limit=5)
    assert any("updated text" == h for h in hits_new)


def test_fts_updates_on_delete(memstore):
    eid = memstore.add_event("delete me please", importance=1.0, type_="event")

    memstore.conn.execute("DELETE FROM events WHERE id=?", (eid,))
    memstore.conn.commit()

    hits = memstore.keyword_search("delete", limit=5)
    assert all("delete me please" != h for h in hits)
