# base/memory/store.py
from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from loguru import logger

from base.database.sqlite import SQLiteConn
from base.memory.scoring import assess_importance
from base.memory.vector_backend import (
    HAVE_QDRANT,  # make sure this exists; see note below
    InMemoryBackend,  # make sure this exists; see note below
    QdrantMemoryBackend,
    VectorBackend,
)
from base.utils.embeddings import get_embedder
from config.config import settings  # OK where your settings live

DB_PATH = Path("memory.db")


class MemoryStore:
    """
    Persistent memory store using SQLite for base storage and an optional vector backend for semantic search.
    Supports key/value facts, events with importance, and both keyword and semantic retrieval.
    """

    def __init__(self, db: SQLiteConn | sqlite3.Connection):
        # 0) normalize DB handle
        self.conn: sqlite3.Connection = db.conn if hasattr(db, "conn") else db  # type: ignore[attr-defined]
        self.conn.row_factory = sqlite3.Row
        
        # Detect SQLite in-memory DBs (common in tests). When using ":memory:",
        # a shared persistent Qdrant collection can leak vectors across tests and
        # make semantic results nondeterministic. Prefer in-process backend there.
        self._is_in_memory_sqlite = False
        try:
            cur = self.conn.execute("PRAGMA database_list")
            rows = cur.fetchall()
            for r in rows:
                name = r["name"] if isinstance(r, sqlite3.Row) else r[1]
                file_ = r["file"] if isinstance(r, sqlite3.Row) else r[2]
                if name == "main":
                    # SQLite reports empty file path for :memory:
                    self._is_in_memory_sqlite = not bool(file_)
                    break
        except Exception:
            # Best effort only; default to False
            self._is_in_memory_sqlite = False

        # a) transient in-memory caches
        self._add_history: list[str] = []
        self._recall_fact: dict[str, Any] = {}
        self._remember_fact: dict[str, Any] = {}

        # 1) state shelves
        self._fts_enabled: bool = False
        self._subscribers: list[Any] = []
        self._bg_threads: list[threading.Thread] = []
        self._bg_threads_lock = threading.Lock()

        # 2) embeddings FIRST (so vector backend can use it)
        self._embedder: Any | None = None
        self._embed_dim: int | None = None
        try:
            # If you want to gate by settings, you still can — this is simple & robust
            self._embedder, self._embed_dim = get_embedder()
            if not self._embed_dim and hasattr(self._embedder, "dim"):
                self._embed_dim = int(self._embedder.dim)
            logger.debug(f"MemoryStore: embedder ready (dim={self._embed_dim})")
        except Exception as e:
            logger.warning(
                f"MemoryStore: no embedder available ({e}); semantic search may be limited"
            )
            self._embedder, self._embed_dim = None, None

        # 3) vector backend selection (AFTER embedder)
        backend_choice = (
            getattr(settings, "aerith_vector_backend", "auto") or "auto"
        ).strip().lower()
        
        if backend_choice not in {"auto", "qdrant", "inmemory"}:
            logger.warning(
                f"MemoryStore: invalid AERITH_VECTOR_BACKEND={backend_choice!r}; using 'auto'"
            )
            backend_choice = "auto"

        self._vector_backend: VectorBackend | None = None
        try:
            # For SQLite ":memory:" DBs (tests), use the in-process backend so vectors
            # do not leak across test cases via a shared Qdrant collection.
            if self._is_in_memory_sqlite:
                self._vector_backend = InMemoryBackend(embedder=self._embedder)
                logger.info("MemoryStore: Vector backend = InMemory (sqlite :memory:)")
            elif backend_choice in ("qdrant", "auto") and HAVE_QDRANT and self._embedder:
                
                qdrant_url = getattr(settings, "qdrant_url", None)
                qdrant_api_key = getattr(settings, "qdrant_api_key", None)
                if isinstance(qdrant_api_key, str):
                    qdrant_api_key = qdrant_api_key.strip() or None
                qdrant_collection = getattr(settings, "qdrant_collection", None) or "events"

                self._vector_backend = QdrantMemoryBackend(
                    embedder=self._embedder,
                    dim=int(self._embed_dim or 384),
                    url=qdrant_url,
                    api_key=qdrant_api_key,
                    collection_name=qdrant_collection,
                )  # type: ignore
                logger.info("MemoryStore: Vector backend = Qdrant")
            else:
                # Always available fallback
                self._vector_backend = InMemoryBackend(embedder=self._embedder)
                logger.info("MemoryStore: Vector backend = InMemory")
        except Exception as e:
            # Backend init failed; fall back to in-memory backend (always available).
            logger.warning(
                f"MemoryStore: vector backend init failed ({e}); falling back to InMemoryBackend."
            )
            try:
                self._vector_backend = InMemoryBackend(embedder=self._embedder)
                logger.info("MemoryStore: Vector backend = InMemory (fallback)")
            except Exception as e2:
                logger.error(
                    f"MemoryStore: InMemory backend init failed; disabling semantic search: {e2}"
                )
                self._vector_backend = None
        
        # 4) schema last (unchanged)
        self._ensure_schema()
        self.init_db()

    @property
    def embedder(self) -> Any | None:
        """Public, read-only view for code that expects `self.embedder`."""
        return self._embedder

    # ... existing schema and fact methods ...

    # ---------- schema ----------
    def _ensure_schema(self) -> None:
        cur = self.conn.cursor()

        # Simple key/value facts
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS facts (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                last_updated TEXT NOT NULL
            )
            """
        )

        # Legacy "memories" table for whole exchanges
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                type TEXT,
                content TEXT,
                response TEXT
            )
            """
        )

        # Events (atomic things worth recalling)
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                ts TEXT NOT NULL,
                importance REAL NOT NULL DEFAULT 0.0,
                type TEXT NOT NULL DEFAULT 'event'
            )
            """
        )

        # Full-text search index linked to events.content
        try:
            cur.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS events_fts
                USING fts5(content, content='events', content_rowid='id')
                """
            )

            # Keep FTS index in sync automatically
            cur.executescript(
                """
                CREATE TRIGGER IF NOT EXISTS events_ai AFTER INSERT ON events BEGIN
                  INSERT INTO events_fts(rowid, content) VALUES (new.id, new.content);
                END;

                CREATE TRIGGER IF NOT EXISTS events_ad AFTER DELETE ON events BEGIN
                  INSERT INTO events_fts(events_fts, rowid, content) VALUES('delete', old.id, old.content);
                END;

                CREATE TRIGGER IF NOT EXISTS events_au AFTER UPDATE ON events BEGIN
                  INSERT INTO events_fts(events_fts, rowid, content) VALUES('delete', old.id, old.content);
                  INSERT INTO events_fts(rowid, content) VALUES (new.id, new.content);
                END;
                """
            )

            self._fts_enabled = True
        except sqlite3.OperationalError as e:
            from loguru import logger

            logger.error(f"FTS init failed: {e}")
            self._fts_enabled = False

        self.conn.commit()

    def _to_safe_fts_query(self, text: str) -> str:
        """
        Convert natural-language text into a safe FTS5 MATCH query.
        Strips punctuation like '?' that can break MATCH parsing.
        """
        tokens = re.findall(r"[A-Za-z0-9_]+", text or "")
        if not tokens:
            return ""
        # Use OR to improve recall for natural language prompts
        return " OR ".join(tokens[:12])

    # ---------- facts ----------
    def upsert_fact(self, key: str, value: str) -> None:
        ts = datetime.utcnow().isoformat()
        self.conn.execute(
            """
            INSERT INTO facts(key, value, last_updated)
            VALUES(?, ?, ?)
            ON CONFLICT(key)
            DO UPDATE SET value=excluded.value, last_updated=excluded.last_updated
            """,
            (key, value, ts),
        )
        self.conn.commit()

    def list_facts(self) -> list[tuple[str, str]]:
        cur = self.conn.execute("SELECT key, value FROM facts ORDER BY key")
        return [(r["key"], r["value"]) for r in cur.fetchall()]

    def add_history(self, text: str) -> None:
        self._add_history.append(text)

    def recall_fact(self, key: str) -> Any:
        return self._recall_fact.get(key)

    def remember_fact(self, key: str, value: Any) -> None:
        self._remember_fact[key] = value

    def forget(self, topic: str) -> int:
        """Delete facts and events containing the given topic string."""
        n1 = (
            self.conn.execute(
                "DELETE FROM facts WHERE key LIKE ? OR value LIKE ?",
                (f"%{topic}%", f"%{topic}%"),
            ).rowcount
            or 0
        )
        n2 = (
            self.conn.execute(
                "DELETE FROM events WHERE content LIKE ?",
                (f"%{topic}%",),
            ).rowcount
            or 0
        )
        self.conn.commit()
        return n1 + n2

    def subscribe(self, callback):
        """
        Register a callback to be called whenever a new event is added.
        Callback signature: fn(content: str, ts: str, type_: str, importance: float).
        """
        self._subscribers.append(callback)

    # ---------- events ----------
    def add_event(self, content: str, importance: float = 0.0, type_: str = "event") -> int:
        """
        Add a new event (memory) to the store.
        This stores the content and metadata in the SQLite DB and triggers an asynchronous embedding
        to store the vector in the vector database (if configured).
        Returns the new event's ID.
        """
        ts = datetime.utcnow().isoformat()
        cur = self.conn.execute(
            "INSERT INTO events(content, ts, importance, type) VALUES(?, ?, ?, ?)",
            (content, ts, importance, type_),
        )
        rowid = cur.lastrowid
        if rowid is None:
            raise RuntimeError("Failed to insert event: lastrowid is None")

        # If FTS indexing is enabled, index this content for keyword search
        if self._fts_enabled:
            try:
                self.conn.execute(
                    "INSERT INTO events_fts(rowid, content) VALUES(?, ?)",
                    (rowid, content),
                )
            except sqlite3.OperationalError:
                # FTS table might be unavailable; disable FTS to avoid further errors
                self._fts_enabled = False

        self.conn.commit()

        # Notify any subscribers about the new event (synchronous callbacks)
        for cb in self._subscribers:
            try:
                cb(content=content, ts=ts, type_=type_, importance=importance)
            except Exception as err:
                logger.error(f"MemoryStore: subscriber callback failed: {err}")

        # Asynchronously embed and store the vector for semantic search (non-blocking)
        backend = self._vector_backend
        if backend is not None:

            def _embed_and_store_vector(
                event_id: int, text: str, timestamp: str, importance_val: float, type_val: str
            ):
                """
                Background task: generate embedding and store in vector DB.
                """
                try:
                    # Prepare metadata payload for vector (timestamp as numeric, importance, type, and ISO string)
                    # Use epoch seconds for timestamp to enable range filtering
                    ts_dt = datetime.fromisoformat(timestamp)  # parse to datetime
                    ts_epoch = int(ts_dt.timestamp())
                    metadata = {
                        "timestamp": ts_epoch,
                        "ts_iso": timestamp,
                        "importance": importance_val,
                        "type": type_val,
                    }
                    # Use QdrantMemoryBackend's add_text to insert vector with given ID and metadata
                    # Retry logic for embedding API call (e.g., OpenAI) in case of transient failures
                    attempts = 0
                    while attempts < 3:
                        try:
                            backend.add_text(text, vector_id=event_id, metadata=metadata)
                            logger.info(
                                f"MemoryStore: Event {event_id} embedded and stored in vector DB."
                            )
                            break  # success
                        except Exception as e:
                            attempts += 1
                            if attempts < 3:
                                logger.warning(
                                    f"MemoryStore: Embedding attempt {attempts} failed for event {event_id}: {e}. Retrying..."
                                )
                                time.sleep(1.0 * attempts)  # exponential backoff delay
                            else:
                                logger.error(
                                    f"MemoryStore: Failed to embed event {event_id} after {attempts} attempts: {e}"
                                )
                except Exception as e:
                    logger.error(
                        f"MemoryStore: Unexpected error in vector embedding thread for event {event_id}: {e}"
                    )

            # Launch the embedding thread (daemon so it won't block program exit)
            t = threading.Thread(
                target=_embed_and_store_vector,
                args=(int(rowid), content, ts, importance, type_),
                daemon=True,
            )
            with self._bg_threads_lock:
                self._bg_threads.append(t)
            t.start()

        return int(rowid)

    def wait_for_background_tasks(self, timeout: float = 5.0) -> None:
        """
        Best-effort join of in-flight background embedding threads.
        Useful for short-lived scripts/tests so Python doesn't exit mid-embed.
        """
        deadline = time.time() + max(0.0, float(timeout))
        while True:
            with self._bg_threads_lock:
                alive = [t for t in self._bg_threads if t.is_alive()]
                self._bg_threads = alive
                threads = list(alive)

            if not threads:
                return

            remaining = deadline - time.time()
            if remaining <= 0:
                return

            # Join each thread briefly, then re-check the list
            per_thread = min(0.25, remaining)
            for t in threads:
                t.join(timeout=per_thread)
                
    def list_events(self, type_: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        """
        Return recent events as dictionaries.
        """
        sql = "SELECT id, content, ts, importance, type FROM events"
        params: list[Any] = []
        if type_:
            sql += " WHERE type = ?"
            params.append(type_)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(int(limit))

        cur = self.conn.execute(sql, params)
        return [dict(r) for r in cur.fetchall()]

    def search(
        self,
        query: str,
        since: str | None = None,
        min_importance: float = 0.0,
        type_: str | None = None,
        limit: int = 5,
    ) -> list[str]:
        """
        Search the memory store for events relevant to the query.

        Supports semantic similarity search via the vector backend (if available), with optional filtering:
        - since: only events on or after this ISO timestamp are considered
        - min_importance: only events with importance >= this value are considered
        - type_: only events of this type are considered

        Falls back to SQLite FTS/LIKE keyword search if semantic search is unavailable or returns no hits.
        """
        results: list[str] = []
        rows: list[sqlite3.Row] | list[Any] = []

        # Search policy:
        # - Open-ended / question-style queries with NO structured filters may use semantic search.
        # - Filtered queries should prefer SQLite keyword search for deterministic behavior.
        use_vector = self._vector_backend is not None

        has_structured_filters = (
            since is not None
            or type_ is not None
            or float(min_importance or 0.0) > 0.0
        )

        if has_structured_filters:
            use_vector = False

        # -----------------------------
        # 1) Semantic / vector search
        # -----------------------------
        if use_vector and self._vector_backend is not None:
            try:
                if isinstance(self._vector_backend, QdrantMemoryBackend):
                    # Qdrant backend handles metadata filtering internally
                    vector_hits = self._vector_backend.search(
                        query,
                        k=limit,
                        since=since,
                        min_importance=min_importance,
                        type_filter=type_,
                    )
                    if vector_hits:
                        return vector_hits[:limit]

                else:
                    # InMemory / FAISS-like backends usually return plain strings and may not support filtering.
                    # Grab extra hits, then apply best-effort post filtering when metadata exists.
                    raw_hits = self._vector_backend.search(query, k=limit * 5)

                    filtered: list[str] = []

                    for hit in raw_hits:
                        text = ""
                        meta: dict[str, Any] = {}

                        # Dict hit: {"text": "...", "metadata": {...}} or {"content": "...", ...}
                        if isinstance(hit, dict):
                            text = str(hit.get("text") or hit.get("content") or "")
                            meta = hit.get("metadata") or hit.get("meta") or {}
                            if not isinstance(meta, dict):
                                meta = {}

                        # Tuple hit: (text, metadata) or (id, text, metadata)
                        elif isinstance(hit, tuple):
                            if len(hit) >= 2 and isinstance(hit[1], str):
                                text = hit[1]
                            elif len(hit) >= 1 and isinstance(hit[0], str):
                                text = hit[0]

                            if len(hit) >= 3 and isinstance(hit[2], dict):
                                meta = hit[2]
                            elif len(hit) >= 2 and isinstance(hit[1], dict):
                                meta = hit[1]

                        # Plain string hit
                        else:
                            text = str(hit)

                        # Best-effort metadata filtering if metadata is available
                        if meta:
                            imp = meta.get("importance", 0.0)
                            try:
                                imp_f = float(imp)
                            except Exception:
                                imp_f = 0.0
                            if min_importance and imp_f < float(min_importance):
                                continue

                            if type_:
                                typ = meta.get("type") or meta.get("type_")
                                if typ is not None and typ != type_:
                                    continue

                            if since:
                                ts_iso = meta.get("ts_iso") or meta.get("ts")
                                if isinstance(ts_iso, str):
                                    try:
                                        ev_time = datetime.fromisoformat(ts_iso.replace("Z", "+00:00"))
                                        since_time = datetime.fromisoformat(since.replace("Z", "+00:00"))
                                        if ev_time < since_time:
                                            continue
                                    except Exception:
                                        pass

                        if text:
                            filtered.append(text)
                        if len(filtered) >= limit:
                            break

                    if filtered:
                        return filtered[:limit]

            except Exception as e:
                logger.error(
                    f"MemoryStore.search: Vector search failed (falling back to keyword search): {e}"
                )

        # -----------------------------
        # 2) Keyword fallback (SQLite)
        # -----------------------------
        try:
            rows = []

            if self._fts_enabled:
                # Sanitize natural-language queries and convert to OR tokens for better recall.
                # Example: "Who did the user meet?" -> "Who OR did OR the OR user OR meet"
                query_str = self._to_safe_fts_query(query)

                if query_str:
                    sql = (
                        "SELECT e.content, e.ts, e.importance, e.type "
                        "FROM events_fts "
                        "JOIN events e ON events_fts.rowid = e.id "
                        "WHERE events_fts MATCH ? "
                        "ORDER BY e.id DESC LIMIT ?"
                    )
                    cur = self.conn.execute(sql, (query_str, limit * 3))
                    rows = cur.fetchall()

            # If FTS is disabled or returned nothing, use LIKE fallback
            if not rows:
                sql = (
                    "SELECT content, ts, importance, type FROM events "
                    "WHERE content LIKE ? ESCAPE '\\' "
                    "ORDER BY id DESC LIMIT ?"
                )
                cur = self.conn.execute(sql, (f"%{query}%", limit * 3))
                rows = cur.fetchall()

        except Exception as e:
            logger.error(f"MemoryStore.search: Database keyword search query failed: {e}")
            return results  # possibly empty

        # -----------------------------
        # 3) Post-filter SQLite rows
        # -----------------------------
        for r in rows:
            content = r["content"]
            ts = r["ts"]
            imp = float(r["importance"] or 0.0)
            typ = r["type"]

            if since is not None:
                try:
                    ev_time = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    since_time = datetime.fromisoformat(since.replace("Z", "+00:00"))
                except Exception:
                    ev_time = None
                    since_time = None

                if ev_time and since_time:
                    if ev_time < since_time:
                        continue
                elif ts < since:
                    continue

            if min_importance and imp < min_importance:
                continue

            if type_ and typ != type_:
                continue

            results.append(content)
            if len(results) >= limit:
                break

        return results
    

    def add_diagnostic_event(
        self,
        *,
        mode: str,
        fix: bool,
        base: str | None,
        diag_output: str,
        issues: list[dict[str, Any]],
        benchmarks: list[dict[str, Any]],
        laggy: bool,
        started_at_iso: str,
        duration_ms: float,
    ) -> None:
        """
        Persist a structured diagnostic record into memory.
        Uses the existing event log mechanism so nothing else in your stack breaks.
        """
        payload = {
            "type": "diagnostic",
            "mode": mode,
            "fix": fix,
            "base": base,
            "tool_output": diag_output[-8000:],  # cap to avoid massive blobs
            "issues": issues,
            "benchmarks": benchmarks,
            "laggy": laggy,
            "started_at": started_at_iso,
            "duration_ms": duration_ms,
            "created_at": datetime.now(UTC).isoformat(),
            "version": 1,
        }
        # Important: keep the same event API your code already uses
        # (content text field + type_ discriminator).
        try:
            self.add_event(
                content=json.dumps(payload, ensure_ascii=False),
                importance=0.0,
                type_="diagnostic",
            )
        except Exception:
            # Graceful fallback: don't crash the run; keep an audit trail.
            try:
                with open("diagnostics.log", "a", encoding="utf-8") as f:
                    f.write(json.dumps({"fallback": True, **payload}) + "\n")
            except Exception:
                pass

    def recent_diagnostics(self, limit: int = 10) -> list[dict[str, Any]]:
        """
        Return the most recent diagnostic events (decoded).
        Falls back to reading diagnostics.log if the event store is unavailable.
        """
        out: list[dict[str, Any]] = []
        try:
            # If you have a native list_events(type_=...) use it here.
            # Otherwise this uses a common content/type_ pattern.
            rows = self.list_events(
                type_="diagnostic", limit=limit
            )  # <-- your existing helper if present
            for r in rows:
                # r["content"] expected to be a JSON string from add_diagnostic_event
                try:
                    out.append(json.loads(r.get("content", "{}")))
                except Exception:
                    continue
            return out
        except Exception:
            # Fallback to file if DB listing fails
            try:
                with open("diagnostics.log", encoding="utf-8") as f:
                    lines = f.readlines()[-limit:]
                for ln in lines:
                    try:
                        out.append(json.loads(ln))
                    except Exception:
                        continue
                return out
            except Exception:
                return out  # empty if nothing works

    def last_diagnostic(self) -> dict[str, Any] | None:
        items = self.recent_diagnostics(limit=1)
        return items[0] if items else None

    def diagnostics_since(self, iso_timestamp: str, limit: int = 50) -> list[dict[str, Any]]:
        """
        Filter diagnostics since a given ISO timestamp.
        """
        try:
            since = datetime.fromisoformat(iso_timestamp.replace("Z", "+00:00"))
        except Exception:
            return self.recent_diagnostics(limit=limit)

        results: list[dict[str, Any]] = []
        for ev in self.recent_diagnostics(limit=limit):
            try:
                ts = datetime.fromisoformat(ev.get("created_at", "").replace("Z", "+00:00"))
                if ts >= since:
                    results.append(ev)
            except Exception:
                continue
        return results

    def maybe_store_text(self, text: str, explicit_type: str | None = None) -> bool:
        score = assess_importance(text)
        if score < settings.importance_threshold:
            return False
        self.add_event(text, importance=score, type_=explicit_type or "event")
        return True

    def prune_events(self) -> int:
        """Prune old, low-importance events according to TTL."""
        ttl = timedelta(days=settings.memory_ttl_days)
        cutoff = (datetime.utcnow() - ttl).isoformat()
        cur = self.conn.execute(
            "DELETE FROM events WHERE ts < ? AND importance < ?",
            (cutoff, float(settings.importance_threshold)),
        )
        self.conn.commit()

        return int(cur.rowcount or 0)

    # ---------- retrieval ----------
    # def keyword_search(self, query, limit):
    #     limit = 5
    #     # Strip commas from the query
    #     query = query.replace(',', '')
    #     # Existing functionality to search using FTS triggers or LIKE fallback
    #     # Add your FTS implementation here
    #     pass
    #     query = query.replace(',', '')  # Strip commas from the query
    #     # Existing functionality for FTS triggers and LIKE fallback goes here
    #     if self._fts_enabled:
    #         try:
    #             sql = f"SELECT content FROM events_fts WHERE events_fts MATCH ? LIMIT {int(limit)}"
    #             cur = self.conn.execute(sql, (query,))
    #             return [r[0] for r in cur.fetchall()]
    #         except sqlite3.OperationalError as e:
    #             from loguru import logger
    #             logger.error(f"FTS search failed, falling back to LIKE: {e}")
    #             self._fts_enabled = False
    #             # fallback to LIKE if FTS errors out
    #     # fallback: LIKE
    #     cur = self.conn.execute(
    #         "SELECT content FROM events WHERE content LIKE ? ORDER BY id DESC LIMIT ?",
    #         (f"%{query}%", limit),
    #     )
    #     # return [dict(r) for r in cur.fetchall()]
    #     return [r[0] for r in cur.fetchall()]

    def keyword_search(self, query: str, limit: int = 5) -> list[str]:
        """
        Search memory for a keyword or phrase.
        Returns a consistent shape: list[str] of content.
        """
        try:
            cur = self.conn.execute(
                "SELECT content FROM memories WHERE content MATCH ? LIMIT ?",
                (query, limit),
            )
            return [str(r[0]) for r in cur.fetchall()]
        except Exception:
            cur = self.conn.execute(
                "SELECT content FROM memories WHERE content LIKE ? LIMIT ?",
                (f"%{query}%", limit),
            )
            return [str(r[0]) for r in cur.fetchall()]

    # ---------- Legacy helpers for backward compatibility ----------

    def _connect_for_compat(self) -> sqlite3.Connection:
        return sqlite3.connect(settings.db_path or str(DB_PATH), check_same_thread=False)

    def init_db(self) -> None:
        with self._connect_for_compat() as conn:
            cur = conn.cursor()
            # run the same schema setup here if needed
            # or simply no-op since _ensure_schema already covers this
            conn.commit()

    def save_memory(self, memory: dict) -> None:
        ts = memory.get("timestamp") or datetime.utcnow().isoformat()
        typ = memory.get("type", "event")
        content = memory.get("content", "")
        response = memory.get("response", "")

        with self._connect_for_compat() as conn:
            conn.execute(
                "INSERT INTO memories(timestamp, type, content, response) VALUES(?, ?, ?, ?)",
                (ts, typ, content, response),
            )
            conn.commit()
