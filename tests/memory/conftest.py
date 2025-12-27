import sqlite3
import pytest

from base.memory.store import MemoryStore


@pytest.fixture()
def memstore():
    # Use an in-memory DB for speed + isolation
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    store = MemoryStore(conn)
    yield store
    conn.close()
