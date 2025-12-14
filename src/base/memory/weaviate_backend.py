# Optional: swap this in via config
from __future__ import annotations

from .vector_backend import VectorBackend


class WeaviateBackend(VectorBackend):
    def __init__(self, client):
        self.client = client

    def index(self, texts: list[str]) -> None:
        # push texts + embeddings to Weaviate schema/class
        pass

    def search(self, query: str, k: int = 5) -> list[str]:
        # query Weaviate for nearest neighbors
        return []
