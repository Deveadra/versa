
from base.memory.store import MemoryStore
from base.database.sqlite import SQLiteConn
from config.config import settings

def recall_memory(query: str) -> str:
    db = SQLiteConn(settings.db_path)
    store = MemoryStore(db.conn)
    results = store.search(query)
    # MemoryStore.search typically returns list[str]; convert to a readable string.
    if isinstance(results, str):
        return results
    if not results:
        return ""
    return "\n".join(results)

def write_memory(content: str) -> str:
    db = SQLiteConn(settings.db_path)
    store = MemoryStore(db.conn)
    store.add_event(content, importance=0.8, type_="agent_note")
    return "Memory stored."
