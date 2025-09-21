
from __future__ import annotations
from typing import Any, List, Sequence
import numpy as np
import faiss

from .vector_backend import VectorBackend

class FAISSBackend(VectorBackend):
    def __init__(self, embedder, dim: int = 384, normalize: bool = True):
        self.embedder = embedder
        self.normalize = normalize
        self.dim = dim
        self.faiss_index: Any = faiss.IndexFlatIP(dim)  # <- Any avoids bogus “missing args”
        self.texts: list[str] = []

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        vecs = self.embedder.encode(list(texts))
        vecs = np.asarray(vecs, dtype=np.float32)   # handle list->ndarray safely
        if self.normalize and vecs.size:
            faiss.normalize_L2(vecs)               # cosine via normalized IP
        return vecs

    def index(self, texts: list[str]) -> None:
        if not self.embedder or not texts:
            return
        vecs = self._encode(texts)
        self.faiss_index.add(vecs,)
        self.texts.extend(texts)

    def search(self, query: str, k: int = 5) -> List[str]:
        if not self.embedder or not self.texts:
            return []
        q = self._encode([query])
        k = min(k, len(self.texts))
        if k <= 0:
            return []
        _, idx = self.faiss_index.search(q, k)
        return [self.texts[i] for i in idx[0] if 0 <= i < len(self.texts)]
