# SPDX-License-Identifier: MIT
# src/base/memory/vector_backend.py
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol


# --- Generic interface --------------------------------------------------------
class VectorBackend(Protocol):
    def index(self, texts: list[str]) -> None: ...
    def add_text(
        self, text: str, vector_id: int | str | None = None, metadata: dict | None = None
    ) -> None: ...
    def search(
        self,
        query: str,
        k: int = 5,
        since: str | None = None,
        min_importance: float = 0.0,
        type_filter: str | None = None,
    ) -> list[str]: ...


# --- Optional Qdrant backend --------------------------------------------------
HAVE_QDRANT = False
try:
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import (
        Distance,
        FieldCondition,
        Filter,
        MatchValue,
        PointStruct,
        Range,
        VectorParams,
    )

    HAVE_QDRANT = True
except Exception:
    QdrantClient = None  # type: ignore

from loguru import logger


class _QdrantMemoryBackendImpl:
    def __init__(
        self,
        embedder,
        dim: int,
        *,
        url: str | None = None,
        api_key: str | None = None,
        collection_name: str = "events",
    ):
        self.embedder = embedder
        self.dim = dim
        self.collection_name = collection_name
        if QdrantClient is None:
            raise ImportError(
                "qdrant-client not installed. Install it or choose a different backend."
            )
        if url:
            logger.info(f"QdrantMemoryBackend: Connecting to remote Qdrant at {url}")
            self.client = QdrantClient(url=url, api_key=api_key)
        else:
            logger.info("QdrantMemoryBackend: Connecting to local Qdrant (localhost:6333)")
            self.client = QdrantClient(host="localhost", port=6333)

        try:
            self.client.get_collection(collection_name=self.collection_name)
        except Exception:
            logger.info(
                f"Creating Qdrant collection '{self.collection_name}' (dim={dim}, metric=Cosine)"
            )
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
            points.append(PointStruct(id=base_id + i, vector=vec.tolist(), payload={"content": t}))
        try:
            self.client.upsert(collection_name=self.collection_name, points=points)
            logger.info(f"QdrantMemoryBackend: Indexed {len(points)} vectors.")
        except Exception as e:
            logger.error(f"QdrantMemoryBackend.index: upsert failed: {e}")

    def add_text(
        self, text: str, vector_id: int | str | None = None, metadata: dict | None = None
    ) -> None:
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

    def search(
        self,
        query: str,
        k: int = 5,
        since: str | None = None,
        min_importance: float = 0.0,
        type_filter: str | None = None,
    ) -> list[str]:
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
                # conds.append(FieldCondition(key="ts_iso", range=Range(gte=since)))
                # Qdrant range filters require numeric types; ignore invalid since filters
                logger.warning(
                    "QdrantMemoryBackend.search: could not parse 'since' as ISO timestamp; ignoring filter."
                )
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


# Export a working Qdrant backend when available; otherwise a safe stub.
if HAVE_QDRANT:
    QdrantMemoryBackend = _QdrantMemoryBackendImpl  # type: ignore[misc]
else:

    class QdrantMemoryBackend:  # type: ignore[no-redef]
        def __init__(self, *a: Any, **kw: Any):
            raise ImportError(
                "qdrant-client not installed. Install it or choose a different backend."
            )


# --- No-deps, in-memory fallback ---------------------------------------------
class InMemoryBackend:
    """
    Minimal, dependency-free backend for development and tests.
    Stores (text, vector) pairs in memory and does cosine search.
    """

    def __init__(self, embedder=None):
        try:
            import numpy as np  # preferred, but optional
        except Exception:
            np = None
        self.np = np
        self.embedder = embedder
        self._rows: list[tuple[str, object, dict]] = []  # (text, vector, metadata)

    def index(self, texts: list[str]) -> None:
        if not texts:
            return
        for t in texts:
            self.add_text(t)

    def add_text(
        self, text: str, vector_id: int | str | None = None, metadata: dict | None = None
    ) -> None:
        if not self.embedder:
            return
        try:
            v = self.embedder.encode([text])[0]
        except Exception:
            return
        self._rows.append((text, v, dict(metadata or {})))

    def _cosine(self, a, b) -> float:
        # NumPy fast path if available
        if self.np is not None:
            denom = (self.np.linalg.norm(a) * self.np.linalg.norm(b)) or 1.0
            return float(self.np.dot(a, b) / denom)
        # Pure-Python fallback
        dot = sum(float(x) * float(y) for x, y in zip(a, b))
        norm_a = sum(float(x) * float(x) for x in a) ** 0.5
        norm_b = sum(float(y) * float(y) for y in b) ** 0.5
        denom = (norm_a * norm_b) or 1.0
        return float(dot / denom)

    def search(
        self,
        query: str,
        k: int = 5,
        since: str | None = None,
        min_importance: float = 0.0,
        type_filter: str | None = None,
        **_filters,
    ) -> list[str]:
        if not self._rows or not self.embedder:
            return []
        try:
            q = self.embedder.encode([query])[0]
        except Exception:
            return []

        # Apply filters using stored metadata (if provided)
        rows = self._rows

        if since:
            try:
                import datetime as _dt
                since_epoch = int(_dt.datetime.fromisoformat(since.replace("Z", "+00:00")).timestamp())
                rows = [r for r in rows if int((r[2].get("timestamp") or 0)) >= since_epoch]
            except Exception:
                pass

        if min_importance and min_importance > 0:
            rows = [r for r in rows if float(r[2].get("importance") or 0.0) >= float(min_importance)]

        if type_filter:
            rows = [r for r in rows if str(r[2].get("type") or "") == str(type_filter)]

        if not rows:
            return []

        sims: list[tuple[float, str]] = [(self._cosine(q, v), text) for (text, v, _m) in rows]
        sims.sort(key=lambda x: x[0], reverse=True)
        return [t for _, t in sims[:k]]


__all__ = ["VectorBackend", "QdrantMemoryBackend", "InMemoryBackend", "HAVE_QDRANT"]
