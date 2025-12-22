from __future__ import annotations

import time
import sqlite3

from base.database.sqlite import SQLiteConn
from base.memory.store import MemoryStore

def test_memory_semantic_search():
    # Set up an in-memory SQLite database for testing
    db_conn = SQLiteConn(":memory:")
    store = MemoryStore(db_conn.conn)
    # Ensure that the OpenAI API key or a local embedding model is configured for embeddings
    assert store._vector_backend is not None, "Vector backend not initialized. Check Qdrant and embedding config."

    # Add some events to memory
    event1_text = "User met Orion at the park."
    event2_text = "Team meeting scheduled for next week."
    id1 = store.add_event(event1_text, importance=30.0, type_="event")
    id2 = store.add_event(event2_text, importance=5.0, type_="event")
    time.sleep(1.0)  # small delay to allow background embedding threads to finish

    # Perform an unfiltered semantic search
    results = store.search("Who did the user meet?", min_importance=0.0)
    print("Search results (semantic):", results)
    assert any("Alice" in res for res in results), "Semantic search failed to retrieve the correct event."

    # Perform a filtered search by keyword and recency (e.g., events since a recent timestamp)
    since_ts = "2025-01-01T00:00:00"
    results_filtered = store.search("meeting", since=since_ts, type_="event")
    print("Search results (filtered):", results_filtered)
    for res in results_filtered:
        # All results should contain 'meeting' (keyword match) and be of type "event"
        assert "meeting" in res.lower()
    # Also ensure that filtering by importance works (e.g., min_importance)
    results_imp = store.search("meeting", min_importance=10.0)
    for res in results_imp:
        # Only the more important meeting (importance 30.0) should appear
        assert "Team meeting" in res  # the lower importance event should be filtered out

    # Clean up (if needed)
    del store
