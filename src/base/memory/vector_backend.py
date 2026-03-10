# SPDX-License-Identifier: MIT
# src/base/memory/vector_backend.py

from __future__ import annotations

import time
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

        points: list[PointStruct] = []
        base_id = int(time.time() * 1000)
        points.extend(
            PointStruct(
                id=base_id + i, vector=vec.tolist(), payload={"content": t}
            )
            for i, (t, vec) in enumerate(zip(texts, vectors, strict=True))
        )
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
        """
        Semantic search against Qdrant with compatibility across qdrant-client versions.

        Expected return shape for MemoryStore integration: list[str] (event contents)
        """
        if not query or self.embedder is None:
            return []

        try:
            qv = self.embedder.encode([query])[0]
            qv = qv.tolist() if hasattr(qv, "tolist") else list(qv)
        except Exception as e:
            logger.error(f"QdrantMemoryBackend.search: embed failed: {e}")
            return []

        # Build server-side filter (when supported) for stable, cheap filtering.
        # We intentionally apply `since` client-side because API / payload typing varies across versions.
        must_conditions = []

        if type_filter:
            try:
                must_conditions.append(
                    FieldCondition(key="type", match=MatchValue(value=type_filter))
                )
            except Exception:
                # If model signatures vary, skip server-side type filter and do client-side filtering
                pass

        try:
            min_imp = float(min_importance or 0.0)
        except Exception:
            min_imp = 0.0

        if min_imp > 0.0:
            try:
                must_conditions.append(FieldCondition(key="importance", range=Range(gte=min_imp)))
            except Exception:
                # If model signatures vary, skip server-side importance filter and do client-side filtering
                pass

        qfilter = Filter(must=must_conditions) if must_conditions else None

        # If we need client-side `since` filtering, fetch extra candidates.
        fetch_k = max(int(k), 1)
        if since:
            fetch_k = max(fetch_k * 3, 10)

        def _clean_kwargs(d: dict[str, Any]) -> dict[str, Any]:
            return {kk: vv for kk, vv in d.items() if vv is not None}

        def _try_call_variants(fn, variants: list[dict[str, Any]]):
            last_err: Exception | None = None
            for kwargs in variants:
                try:
                    return fn(**_clean_kwargs(kwargs))
                except TypeError as e:
                    last_err = e
                    continue
                except Exception as e:
                    msg = str(e)
                    # qdrant-client often raises custom exceptions for unknown kwargs
                    if (
                        "Unknown arguments" in msg
                        or "unexpected keyword" in msg
                        or "got an unexpected keyword" in msg
                    ):
                        last_err = e
                        continue
                    raise
            if last_err:
                raise last_err
            return None

        try:
            points = None

            # --- Preferred path: newer qdrant-client API (`query_points`) ---
            if hasattr(self.client, "query_points"):
                query_points_variants = [
                    {
                        "collection_name": self.collection_name,
                        "query": qv,
                        "query_filter": qfilter,
                        "limit": fetch_k,
                        "with_payload": True,
                        "with_vectors": False,
                    },
                    {
                        "collection_name": self.collection_name,
                        "query": qv,
                        "filter": qfilter,  # some versions use `filter`
                        "limit": fetch_k,
                        "with_payload": True,
                        "with_vectors": False,
                    },
                    {
                        "collection_name": self.collection_name,
                        "query": qv,
                        "query_filter": qfilter,
                        "limit": fetch_k,
                        "with_payload": True,
                    },
                    {
                        "collection_name": self.collection_name,
                        "query": qv,
                        "filter": qfilter,
                        "limit": fetch_k,
                        "with_payload": True,
                    },
                    {
                        "collection_name": self.collection_name,
                        "query": qv,
                        "limit": fetch_k,
                    },
                ]

                resp = _try_call_variants(self.client.query_points, query_points_variants)
                raw_points = getattr(resp, "points", resp)
                points = list(raw_points or [])

            # --- Fallback path: older / alternate `search(...)` API ---
            if points is None and hasattr(self.client, "search"):
                search_variants = [
                    {
                        "collection_name": self.collection_name,
                        "query_vector": qv,
                        "query_filter": qfilter,
                        "limit": fetch_k,
                        "with_payload": True,
                    },
                    {
                        "collection_name": self.collection_name,
                        "query_vector": qv,
                        "filter": qfilter,
                        "limit": fetch_k,
                        "with_payload": True,
                    },
                    {
                        "collection_name": self.collection_name,
                        "query": qv,  # some versions use `query` instead of `query_vector`
                        "query_filter": qfilter,
                        "limit": fetch_k,
                        "with_payload": True,
                    },
                    {
                        "collection_name": self.collection_name,
                        "query": qv,
                        "filter": qfilter,
                        "limit": fetch_k,
                        "with_payload": True,
                    },
                    {
                        "collection_name": self.collection_name,
                        "query": qv,
                        "limit": fetch_k,
                    },
                ]

                resp = _try_call_variants(self.client.search, search_variants)
                points = list(resp or [])

            if points is None:
                return []

            out: list[str] = []

            for p in points:
                # Support object-style scored points and dict-like points
                payload = getattr(p, "payload", None)
                if payload is None and isinstance(p, dict):
                    payload = p.get("payload")
                if not isinstance(payload, dict):
                    payload = {}

                content = payload.get("content")
                if not isinstance(content, str) or not content:
                    continue

                # Client-side fallback filters (in case server-side filter was unsupported)
                if type_filter:
                    typ = payload.get("type") or payload.get("type_")
                    if typ is not None and typ != type_filter:
                        continue

                if min_imp > 0.0:
                    try:
                        imp = float(payload.get("importance", 0.0) or 0.0)
                    except Exception:
                        imp = 0.0
                    if imp < min_imp:
                        continue

                if since:
                    ts_iso = payload.get("ts_iso") or payload.get("ts")
                    if isinstance(ts_iso, str):
                        # ISO strings generally compare lexicographically if normalized.
                        # If formatting differs, we skip strict rejection rather than false-negative.
                        try:
                            if ts_iso < since:
                                continue
                        except Exception:
                            pass

                out.append(content)
                if len(out) >= k:
                    break

            return out

        except Exception as e:
            logger.error(f"QdrantMemoryBackend.search: failed: {e}")
            return []


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
            import numpy as np  # noqa: PLC0415
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
        dot = sum(float(x) * float(y) for x, y in zip(a, b, strict=True))
        norm_a = sum(float(x)**2 for x in a)**0.5
        norm_b = sum(float(y)**2 for y in b)**0.5
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
