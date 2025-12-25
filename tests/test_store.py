from base.database.sqlite import SQLiteConn
from base.memory.store import MemoryStore


def test_store_roundtrip(tmp_path):
    db = SQLiteConn(tmp_path / "t.db")
    store = MemoryStore(db)
    store.upsert_fact("sister_name", "Alice")
    facts = dict(store.list_facts())
    assert facts["sister_name"] == "Alice"
    rid = store.add_event("Met Alice at cafe")
    assert isinstance(rid, int)
