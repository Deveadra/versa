
from __future__ import annotations
from typing import List
import numpy as np
import faiss

from assistant.config import settings
from base.memory.store import MemoryStore
from base.memory.store import keyword_search
from .vector_backend import VectorBackend

class Retriever:
    def __init__(self, store: MemoryStore, embedder=None, dim: int = 384):
        self.store = store
        self.embedder = embedder # Embeddings() or None
        self.backend = backend
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim) if embedder else None
        self.texts: list[str] = []


    def index_texts(self, texts: list[str]):
        if not self.embedder:
         return
        vecs = self.embedder.encode(texts).astype("float32")
        self.index.add(vecs)
        self.texts.extend(texts)
        if self.backend:
            self.backend.index(texts)


    def search(self, query: str, k: int = 5) -> List[str]:
        if self.embedder and self.index and len(self.texts) > 0:
        # if self.backend:
            results = self.backend.search(query, k)
            q = self.embedder.encode([query]).astype("float32")
            sims, idx = self.index.search(q, k)
            if results:
                return results
            return [self.texts[i] for i in idx[0] if i < len(self.texts)]
            # fallback keyword
        return self.store.keyword_search(query, limit=k)
