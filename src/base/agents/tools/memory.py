
from base.memory.store import MemoryStore
from base.database.sqlite import SQLiteConn
from config.config import settings

def recall_memory(query: str) -> str:
    db = SQLiteConn(settings.db_path)
    store = MemoryStore(db.conn)
    return store.search(query)

def write_memory(content: str) -> str:
    db = SQLiteConn(settings.db_path)
    store = MemoryStore(db.conn)
    store.add_event(content, importance=0.8, type_="agent_note")
    return "Memory stored."
