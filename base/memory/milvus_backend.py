
# Optional: swap this in via config
from __future__ import annotations
from typing import List


from .vector_backend import VectorBackend


class MilvusBackend(VectorBackend):
def __init__(self, client):
self.client = client


def index(self, texts: list[str]) -> None:
# insert embeddings into Milvus collection
pass


def search(self, query: str, k: int = 5) -> List[str]:
# query Milvus collection
return []
