from __future__ import annotations

from ..memory.store import MemoryStore
from .extract import process_text
from .store import KGStore


class KGIntegrator:
    def __init__(self, store: MemoryStore, kg_store: KGStore):
        self.store = store
        self.kg_store = kg_store

    def ingest_event(self, text: str):
        process_text(self.kg_store, text)
