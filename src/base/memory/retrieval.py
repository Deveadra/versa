from __future__ import annotations

from typing import Any

import faiss

from base.memory.store import MemoryStore
from .vector_backend import VectorBackend


class VectorRetriever:
    def __init__(
        self,
        store: MemoryStore,
        embedder: Any | None = None,
        backend: VectorBackend | None = None,
        dim: int = 384,
    ):
        self.store = store
        self.embedder = embedder  # Embeddings() or None
        self.backend = backend
        self.dim = dim
        self.index: Any = faiss.IndexFlatIP(dim) if embedder else None
        self.texts: list[str] = []

    def index_texts(self, texts: list[str]):
        # Vector index (if we have an embedder)
        if self.embedder and self.index is not None:
            vecs = self.embedder.encode(texts).astype("float32")
            self.index.add(vecs)
            self.texts.extend(texts)

        # Optional secondary backend
        if self.backend:
            self.backend.index(texts)

    def search(self, query: str, k: int = 5) -> list[str]:
        # Try backend first if available
        if self.backend:
            results = self.backend.search(query, k)
            if results:
                return results

        # Then try local vector index
        if self.embedder and self.index is not None and len(self.texts) > 0:
            q = self.embedder.encode([query]).astype("float32")
            # cap k to number of indexed texts to avoid out-of-range
            top_k = min(k, len(self.texts))
            _, idx = self.index.search(q, top_k)
            return [self.texts[i] for i in idx[0] if i < len(self.texts)]

        # Fallback to keyword search
        return self.store.keyword_search(query, limit=k)
        # rows = self.store.keyword_search(query, limit=k)
        # if not rows:
        #     return []
        # # store.keyword_search() may return list[dict] or list[str] depending on implementation.
        # if isinstance(rows[0], dict):
        #     out: list[str] = []
        #     for r in cast(list[dict[str, Any]], rows):
        #         content = r.get("content")
        #         if content:
        #             out.append(str(content))
        #     return out[:k]
        # return cast(list[str], rows)[:k]
