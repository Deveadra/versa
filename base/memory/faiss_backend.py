
from __future__ import annotations
from typing import List
import numpy as np
import faiss


from .vector_backend import VectorBackend


class FAISSBackend(VectorBackend):
  def __init__(self, embedder, dim: int = 384):
    self.embedder = embedder
    self.index = faiss.IndexFlatIP(dim)
    self.texts: list[str] = []


  def index(self, texts: list[str]) -> None:
    if not self.embedder:
      return
    vecs = self.embedder.encode(texts).astype("float32")
    self.index.add(vecs)
    self.texts.extend(texts)


  def search(self, query: str, k: int = 5) -> List[str]:
    if not self.embedder or not self.texts:
      return []
    q = self.embedder.encode([query]).astype("float32")
    sims, idx = self.index.search(q, k)
    return [self.texts[i] for i in idx[0] if i < len(self.texts)]