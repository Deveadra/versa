from __future__ import annotations

import re

from base.database.sqlite import SQLiteConn
from base.memory.store import MemoryStore


class FakeVectorBackend:
    """
    Deterministic, local vector backend stub.
    Stores texts + metadata and does simple token matching.
    """

    def __init__(self) -> None:
        self.items: list[dict] = []

    def index(self, texts: list[str]) -> None:
        for text in texts:
            self.add_text(text)

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
        tokens = [tok for tok in re.findall(r"[A-Za-z0-9_]+", query.lower()) if len(tok) >= 4]
        results: list[str] = []

        for item in self.items:
            metadata = item["metadata"]
            importance = float(metadata.get("importance", 0.0))
            item_type = metadata.get("type")

            if importance < float(min_importance):
                continue
            if type_filter is not None and item_type != type_filter:
                continue

            text = item["text"].lower()
            if not tokens or any(token in text for token in tokens):
                results.append(item["text"])

        return results[:k]


def _make_store_with_fake_backend() -> tuple[MemoryStore, FakeVectorBackend]:
    db_conn = SQLiteConn(":memory:")
    store = MemoryStore(db_conn.conn)
    fake = FakeVectorBackend()
    store._vector_backend = fake  # type: ignore[attr-defined]
    return store, fake


def test_memory_semantic_search() -> None:
    store, _fake = _make_store_with_fake_backend()

    store.add_event(
        "User met Orion at the park.",
        importance=30.0,
        type_="event",
        vector_write="sync",
    )
    store.add_event(
        "Team meeting scheduled for next week.",
        importance=5.0,
        type_="event",
        vector_write="sync",
    )
    store.add_event(
        "Reminder: buy groceries.",
        importance=1.0,
        type_="note",
        vector_write="sync",
    )

    results = store.search("Who did the user meet?", min_importance=0.0)
    assert any("Orion" in result for result in results)

    important_results = store.search("meeting", min_importance=10.0)
    assert all("Team meeting" not in result for result in important_results)

    notes = store.search("groceries", type_="note")
    assert any("groceries" in result.lower() for result in notes)


def test_add_event_sync_vector_write_is_immediate_and_thread_free() -> None:
    store, fake = _make_store_with_fake_backend()

    rowid = store.add_event(
        "System generated summary event.",
        importance=0.2,
        type_="dream_summary",
        vector_write="sync",
    )

    assert rowid > 0
    assert [item["text"] for item in fake.items] == ["System generated summary event."]
    assert not any(thread.is_alive() for thread in store._bg_threads)


def test_add_event_can_skip_vector_write_for_system_events() -> None:
    store, fake = _make_store_with_fake_backend()

    rowid = store.add_event(
        '{"kind":"diagnostic","status":"ok"}',
        importance=0.0,
        type_="diagnostic",
        vector_write="off",
    )

    assert rowid > 0
    events = store.list_events(type_="diagnostic", limit=1)
    assert len(events) == 1
    assert events[0]["content"] == '{"kind":"diagnostic","status":"ok"}'
    assert fake.items == []
    assert not any(thread.is_alive() for thread in store._bg_threads)
