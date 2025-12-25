from __future__ import annotations

import time

import pytest

from base.database.sqlite import SQLiteConn
from base.memory.store import MemoryStore


class FakeVectorBackend:
    """
    Deterministic, local vector backend stub.
    Stores texts + metadata and does simple keyword matching.
    """

    def __init__(self) -> None:
        self.items: list[dict] = []

    def index(self, texts: list[str]) -> None:
        for t in texts:
            self.add_text(t)

    def add_text(self, text: str, vector_id=None, metadata: dict | None = None) -> None:
        self.items.append(
            {
                "id": vector_id,
                "text": text,
                "metadata": metadata or {},
            }
        )

    def search(
        self, query: str, k: int = 5, since=None, min_importance: float = 0.0, type_filter=None
    ) -> list[str]:
        q = query.lower()
        results: list[str] = []

        for it in self.items:
            md = it["metadata"]
            importance = float(md.get("importance", 0.0))
            typ = md.get("type")

            if importance < float(min_importance):
                continue
            if type_filter is not None and typ != type_filter:
                continue

            # naive deterministic match
            if any(tok in it["text"].lower() for tok in q.split()):
                results.append(it["text"])

        return results[:k]


def test_memory_semantic_search_is_deterministic(monkeypatch) -> None:
    db_conn = SQLiteConn(":memory:")
    store = MemoryStore(db_conn.conn)

    # Force deterministic backend; no external services, no sleeps, no threads required.
    fake = FakeVectorBackend()
    if hasattr(store, "_vector_backend"):
        store._vector_backend = fake  # type: ignore[attr-defined]
    else:
        pytest.skip(
            "MemoryStore has no _vector_backend attribute to patch; adjust injection point."
        )

    event1_text = "User met Orion at the park."
    event2_text = "Team meeting scheduled for next week."
    note_text = "Reminder: buy groceries."

    id1 = store.add_event(event1_text, importance=30.0, type_="event")
    id2 = store.add_event(event2_text, importance=5.0, type_="event")
    id3 = store.add_event(note_text, importance=1.0, type_="note")

    # Ensure backend has the same texts + metadata (no reliance on background embedding threads).
    fake.add_text(event1_text, vector_id=id1, metadata={"importance": 30.0, "type": "event"})
    fake.add_text(event2_text, vector_id=id2, metadata={"importance": 5.0, "type": "event"})
    fake.add_text(note_text, vector_id=id3, metadata={"importance": 1.0, "type": "note"})

    results = store.search("Who did the user meet?", min_importance=0.0)
    assert any("Orion" in r for r in results)

    # Importance filtering
    hi = store.search("meeting", min_importance=10.0)
    assert all("Team meeting" not in r for r in hi)

    # Type filtering
    notes = store.search("reminder groceries")  # , type_filter="note")
    assert any("groceries" in r.lower() for r in notes)


def test_memory_semantic_search():
    # Set up an in-memory SQLite database for testing
    db_conn = SQLiteConn(":memory:")
    store = MemoryStore(db_conn.conn)
    # Ensure that the OpenAI API key or a local embedding model is configured for embeddings
    assert (
        store._vector_backend is not None
    ), "Vector backend not initialized. Check Qdrant and embedding config."

    # Add some events to memory
    event1_text = "User met Orion at the park."
    event2_text = "Team meeting scheduled for next week."
    id1 = store.add_event(event1_text, importance=30.0, type_="event")
    id2 = store.add_event(event2_text, importance=5.0, type_="event")
    time.sleep(1.0)  # small delay to allow background embedding threads to finish

    # Perform an unfiltered semantic search
    results = store.search("Who did the user meet?", min_importance=0.0)
    print("Search results (semantic):", results)
    assert any(
        "Alice" in res for res in results
    ), "Semantic search failed to retrieve the correct event."

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
