from __future__ import annotations

from typing import Protocol, Sequence

# --- Generic interface --------------------------------------------------------
class VectorBackend(Protocol):
    def index(self, texts: list[str]) -> None: ...
    def add_text(self, text: str, vector_id: int | str | None = None, metadata: dict | None = None) -> None: ...
    def search(
        self, query: str, k: int = 5,
        since: str | None = None,
        min_importance: float = 0.0,
        type_filter: str | None = None,
    ) -> list[str]: ...

# --- Optional Qdrant backend --------------------------------------------------
HAVE_QDRANT = False
try:
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import (
        Distance, VectorParams, PointStruct, Filter,
        FieldCondition, Range, MatchValue,
    )
    HAVE_QDRANT = True
except Exception:
    # Keep import failure non-fatal; we'll provide a stub class below.
    QdrantClient = None  # type: ignore

from loguru import logger

if HAVE_QDRANT:
    class QdrantMemoryBackend:
        def __init__(self, embedder, dim: int, *, url: str | None = None,
                     api_key: str | None = None, collection_name: str = "events"):
            self.embedder = embedder
            self.dim = dim
            self.collection_name = collection_name
            if url:
                logger.info(f"QdrantMemoryBackend: Connecting to remote Qdrant at {url}")
                self.client = QdrantClient(url=url, api_key=api_key)
            else:
                logger.info("QdrantMemoryBackend: Connecting to local Qdrant (localhost:6333)")
                self.client = QdrantClient(host="localhost", port=6333)

            try:
                self.client.get_collection(collection_name=self.collection_name)
            except Exception:
                logger.info(f"Creating Qdrant collection '{self.collection_name}' (dim={dim}, metric=Cosine)")
                self.client.recreate_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
                )

        def index(self, texts: list[str]) -> None:
            if not texts or self.embedder is None:
                return
            try:
                vectors = self.embedder.encode(texts)
            except Exception as e:
                logger.error(f"QdrantMemoryBackend.index: Embedding batch failed: {e}")
                return

            points: list[PointStruct] = []
            import time
            base_id = int(time.time() * 1000)
            for i, (t, vec) in enumerate(zip(texts, vectors)):
                points.append(
                    PointStruct(id=base_id + i, vector=vec.tolist(), payload={"content": t})
                )
            try:
                self.client.upsert(collection_name=self.collection_name, points=points)
                logger.info(f"QdrantMemoryBackend: Indexed {len(points)} vectors.")
            except Exception as e:
                logger.error(f"QdrantMemoryBackend.index: upsert failed: {e}")

        def add_text(self, text: str, vector_id: int | str | None = None, metadata: dict | None = None) -> None:
            payload = dict(metadata or {})
            payload.setdefault("content", text)
            try:
                vec = self.embedder.encode([text])[0]
            except Exception as e:
                logger.error(f"QdrantMemoryBackend.add_text: embed failed: {e}")
                return
            point = PointStruct(id=vector_id or 0, vector=vec.tolist(), payload=payload)
            try:
                self.client.upsert(collection_name=self.collection_name, points=[point])
            except Exception as e:
                logger.error(f"QdrantMemoryBackend.add_text: upsert failed: {e}")

        def search(self, query: str, k: int = 5,
                   since: str | None = None, min_importance: float = 0.0, type_filter: str | None = None) -> list[str]:
            try:
                query_vec = self.embedder.encode([query])[0]
            except Exception as e:
                logger.error(f"QdrantMemoryBackend.search: embed failed: {e}")
                return []
            q_filter: Filter | None = None
            conds = []
            if since:
                try:
                    import datetime as _dt
                    ts = int(_dt.datetime.fromisoformat(since.replace("Z", "+00:00")).timestamp())
                    conds.append(FieldCondition(key="timestamp", range=Range(gte=ts)))
                except Exception:
                    conds.append(FieldCondition(key="ts_iso", range=Range(gte=since)))
            if min_importance and min_importance > 0:
                conds.append(FieldCondition(key="importance", range=Range(gte=min_importance)))
            if type_filter:
                conds.append(FieldCondition(key="type", match=MatchValue(value=type_filter)))
            if conds:
                q_filter = Filter(must=conds)

            try:
                results = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vec.tolist(),
                    limit=k,
                    with_payload=True,
                    filter=q_filter,
                )
            except Exception as e:
                logger.error(f"QdrantMemoryBackend.search: failed: {e}")
                return []
            out: list[str] = []
            for r in results or []:
                if r.payload and "content" in r.payload:
                    out.append(r.payload["content"])
            return out
else:
    class QdrantMemoryBackend:  # stub so imports don’t crash
        def __init__(self, *a, **kw):
            raise ImportError("qdrant-client not installed. Install it or choose a different backend.")

# --- No-deps, in-memory fallback ---------------------------------------------
class InMemoryBackend:
    """
    Minimal, dependency-free backend for development and tests.
    Stores (text, vector) pairs in memory and does cosine search.
    """
    def __init__(self, embedder):
        import numpy as np  # std in scientific stacks; if missing, we’ll degrade to dot product
        self.np = np
        self.embedder = embedder
        self._rows: list[tuple[str, "np.ndarray"]] = []

    def index(self, texts: list[str]) -> None:
        for t in texts:
            self.add_text(t)

    def add_text(self, text: str, vector_id: int | str | None = None, metadata: dict | None = None) -> None:
        try:
            v = self.embedder.encode([text])[0]
        except Exception:
            return
        self._rows.append((text, v))

    def search(self, query: str, k: int = 5,
               since: str | None = None,
               min_importance: float = 0.0,
               type_filter: str | None = None) -> list[str]:
        if not self._rows:
            return []
        try:
            q = self.embedder.encode([query])[0]
        except Exception:
            return []
        sims: list[tuple[float, str]] = []
        for text, v in self._rows:
            denom = (self.np.linalg.norm(q) * self.np.linalg.norm(v)) or 1.0
            sims.append((float(self.np.dot(q, v) / denom), text))
        sims.sort(key=lambda x: x[0], reverse=True)
        return [t for _, t in sims[:k]]

__all__ = ["VectorBackend", "QdrantMemoryBackend", "InMemoryBackend", "HAVE_QDRANT"]
