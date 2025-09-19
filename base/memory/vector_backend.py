
from __future__ import annotations
from typing import List, Protocol


class VectorBackend(Protocol):
  def index(self, texts: list[str]) -> None: ...
  def search(self, query: str, k: int = 5) -> List[str]: ...