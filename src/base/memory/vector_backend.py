# SPDX-License-Identifier: MIT
# src/base/memory/vector_backend.py
from __future__ import annotations

from typing import Any, Protocol

from loguru import logger

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
_QDRANT_IMPORT_ERROR: str | None = None

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
except Exception as e:
    QdrantClient = None  # type: ignore
    _QDRANT_IMPORT_ERROR = repr(e)


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
        if not HAVE_QDRANT or QdrantClient is None:
            raise ImportError(
                f"qdrant-client not installed or failed to import ({_QDRANT_IMPORT_ERROR}). "
                "Install it or choose a different backend."
            )

        self.embedder = embedder
        self.dim = dim
        self.collection_name = collection_name

        if url:
            logger.info(f"QdrantMemoryBackend: Connecting to Qdrant at {url}")
            if api_key:
                self.client = QdrantClient(url=url, api_key=api_key)
            else:
                self.client = QdrantClient(url=url)
        else:
            logger.info("QdrantMemoryBackend: Connecting to local Qdrant (localhost:6333)")
            if api_key:
                self.client = QdrantClient(host="localhost", port=6333, api_key=api_key)
            else:
                self.client = QdrantClient(host="localhost", port=6333)
        # Ensure collection exists (DO NOT recreate: that can wipe data)
        try:
            self.client.get_collection(collection_name=self.collection_name)
        except Exception:
            logger.info(
                f"Creating Qdrant collection '{self.collection_name}' (dim={self.dim}, metric=Cosine)"
            )
            self.client.create_collection(
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

        import time

        points: list[PointStruct] = []
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

        import time

        pid = vector_id if vector_id is not None else int(time.time() * 1000)
        point = PointStruct(id=pid, vector=vec.tolist(), payload=payload)

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
                logger.warning(
                    "QdrantMemoryBackend.search: could not parse 'since' as ISO timestamp; ignoring filter."
                )

        if min_importance and min_importance > 0:
            conds.append(FieldCondition(key="importance", range=Range(gte=min_importance)))

        if type_filter:
            conds.append(FieldCondition(key="type", match=MatchValue(value=type_filter)))

        if conds:
            q_filter = Filter(must=conds)

        def _call_search() -> Any:
            # Build common kwargs (only include filter arg if we actually have one)
            base_kwargs: dict[str, Any] = {
                "collection_name": self.collection_name,
                "limit": k,
                "with_payload": True,
            }

            # Vector arg name varies across versions
            vec_list = query_vec.tolist() if hasattr(query_vec, "tolist") else list(query_vec)

            # 1) Try modern/expected client.search(...)
            if hasattr(self.client, "search"):
                # Some clients want query_vector=..., some accept filter=..., some query_filter=...
                # Try a couple safe permutations.
                attempts: list[dict[str, Any]] = []

                kw = dict(base_kwargs)
                kw["query_vector"] = vec_list
                if q_filter is not None:
                    kw["query_filter"] = q_filter
                attempts.append(kw)

                kw = dict(base_kwargs)
                kw["query_vector"] = vec_list
                if q_filter is not None:
                    kw["filter"] = q_filter
                attempts.append(kw)

                last_err: Exception | None = None
                for kw in attempts:
                    try:
                        return self.client.search(**kw)
                    except TypeError as e:
                        last_err = e
                        continue
                if last_err is not None:
                    raise last_err

            # 2) Fallback: client.query_points(...)
            if hasattr(self.client, "query_points"):
                attempts = []

                # variant A
                kw = dict(base_kwargs)
                kw["query_vector"] = vec_list
                if q_filter is not None:
                    kw["query_filter"] = q_filter
                attempts.append(kw)

                # variant B (some versions call it `query`)
                kw = dict(base_kwargs)
                kw["query"] = vec_list
                if q_filter is not None:
                    kw["query_filter"] = q_filter
                attempts.append(kw)

                # filter param name variant
                kw = dict(base_kwargs)
                kw["query_vector"] = vec_list
                if q_filter is not None:
                    kw["filter"] = q_filter
                attempts.append(kw)

                last_err = None
                for kw in attempts:
                    try:
                        return self.client.query_points(**kw)
                    except TypeError as e:
                        last_err = e
                        continue
                if last_err is not None:
                    raise last_err

            raise AttributeError(
                "Qdrant client has neither search() nor query_points(); incompatible qdrant-client."
            )

        try:
            raw = _call_search()
        except Exception as e:
            logger.error(f"QdrantMemoryBackend.search: failed: {e}")
            return []

        # Normalize response shape
        scored = raw
        if hasattr(scored, "points"):
            scored = getattr(scored, "points")
        elif hasattr(scored, "result"):
            scored = getattr(scored, "result")

        out: list[str] = []
        for r in scored or []:
            payload = getattr(r, "payload", None)
            if payload and isinstance(payload, dict) and "content" in payload:
                out.append(payload["content"])
        return out

class QdrantMemoryBackendStub:
    def __init__(self, *a: Any, **kw: Any):
        raise ImportError(
            f"qdrant-client not installed or failed to import ({_QDRANT_IMPORT_ERROR}). "
            "Install it or choose a different backend."
        )


# Export the correct backend at runtime
if HAVE_QDRANT:
    class QdrantMemoryBackend(_QdrantMemoryBackendImpl):
        pass
else:
    class QdrantMemoryBackend(QdrantMemoryBackendStub):
        pass


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
        self._rows: list[tuple[str, object]] = []  # (text, vector)

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
        self._rows.append((text, v))

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
        sims: list[tuple[float, str]] = [(self._cosine(q, v), text) for text, v in self._rows]
        sims.sort(key=lambda x: x[0], reverse=True)
        return [t for _, t in sims[:k]]


__all__ = ["VectorBackend", "QdrantMemoryBackend", "InMemoryBackend", "HAVE_QDRANT"]