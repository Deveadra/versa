# base/memory/store.py
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time


from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from base.database.sqlite import SQLiteConn
from base.memory.scoring import assess_importance
from base.utils.embeddings import get_embedder
from base.memory.vector_backend import QdrantMemoryBackend, VectorBackend, InMemoryBackend, HAVE_QDRANT
import os
from config.config import settings
from loguru import logger

# ... (existing MemoryStore class definition continues)
DB_PATH = Path("memory.db")

class MemoryStore:
    """
    Persistent memory store using SQLite for base storage and an optional vector backend for semantic search.
    Supports key/value facts, events with importance, and both keyword and semantic retrieval.
    """
    
    def __init__(self, db: SQLiteConn | sqlite3.Connection):
        backend_choice = os.getenv("ULTRON_VECTOR_BACKEND", "auto").lower()
        
        # Normalize to a raw sqlite3 connection
        self.conn: sqlite3.Connection = db.conn if hasattr(db, "conn") else db  # type: ignore[attr-defined]
        self.conn.row_factory = sqlite3.Row
        self._fts_enabled = False
        self._ensure_schema()
        self.init_db()
        self._subscribers: list[Any] = []

        # Vector search components
        self._vector_backend: VectorBackend # | None = None
        if backend_choice in ("qdrant", "auto") and HAVE_QDRANT:
            # If you have settings for remote Qdrant, pass them here; otherwise it will try localhost:6333
            self._vector_backend = QdrantMemoryBackend(embedder=self.embedder, dim=self.embedder.dim)
        else:
            # No dependency path — always available
            self._vector_backend = InMemoryBackend(embedder=self.embedder)
            
        self._embedder: Any | None = None
        self._embed_dim: int | None = None
        # Initialize semantic vector backend if configured (requires embeddings and Qdrant)
        try:
            # Only initialize if an embeddings provider is set (and OpenAI API key if using OpenAI)
            provider = settings.embeddings_provider or ""
            if provider and (provider.lower() == "openai" or provider.lower() == "sentence_transformers"):
                # Get the embedding model (this may load a model or verify API key)
                self._embedder, self._embed_dim = get_embedder()
            if self._embedder and self._embed_dim:
                # Instantiate Qdrant backend (remote or local based on config)
                self._vector_backend = QdrantMemoryBackend(
                    self._embedder,
                    dim=self._embed_dim,
                    url=settings.qdrant_url,
                    api_key=settings.qdrant_api_key,
                    collection_name=settings.qdrant_collection or "events",
                )
                logger.info(f"MemoryStore: Vector backend enabled (collection '{settings.qdrant_collection or 'events'}').")
        except ImportError as e:
            # Qdrant client not installed or embedder not available; proceed without vector backend
            logger.warning(f"MemoryStore: Vector backend not initialized ({e}). Semantic search disabled.")
            self._vector_backend = None
        except Exception as e:
            # If Qdrant connection or embedder init failed, disable vector backend but continue
            logger.error(f"MemoryStore: Failed to initialize vector backend: {e}")
            self._vector_backend = None

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
        if self._vector_backend:
            def _embed_and_store_vector(event_id: int, text: str, timestamp: str, importance_val: float, type_val: str):
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
                        "type": type_val
                    }
                    # Use QdrantMemoryBackend's add_text to insert vector with given ID and metadata
                    # Retry logic for embedding API call (e.g., OpenAI) in case of transient failures
                    attempts = 0
                    while attempts < 3:
                        try:
                            self._vector_backend.add_text(text, vector_id=event_id, metadata=metadata)
                            logger.info(f"MemoryStore: Event {event_id} embedded and stored in vector DB.")
                            break  # success
                        except Exception as e:
                            attempts += 1
                            if attempts < 3:
                                logger.warning(f"MemoryStore: Embedding attempt {attempts} failed for event {event_id}: {e}. Retrying...")
                                time.sleep(1.0 * attempts)  # exponential backoff delay
                            else:
                                logger.error(f"MemoryStore: Failed to embed event {event_id} after {attempts} attempts: {e}")
                except Exception as e:
                    logger.error(f"MemoryStore: Unexpected error in vector embedding thread for event {event_id}: {e}")

            # Launch the embedding thread (daemon so it won't block program exit)
            threading.Thread(
                target=_embed_and_store_vector,
                args=(int(rowid), content, ts, importance, type_),
                daemon=True
            ).start()

        return int(rowid)


    def search(self, query: str, since: str | None = None, min_importance: float = 0.0, type_: str | None = None, limit: int = 5) -> list[str]:
        """
        Search the memory store for events relevant to the query.
        Supports semantic similarity search via the vector backend (if available), with optional filtering:
          - since: an ISO timestamp string; if provided, only events on or after this time are considered.
          - min_importance: if > 0, only events with importance >= this value are considered.
          - type_: if provided, only events of this type are considered (e.g. "dream_summary", "agent_step", "event").
        Results are sorted by semantic relevance combined with importance.
        Falls back to keyword search if semantic search is unavailable.
        """
        results: list[str] = []
        # If semantic vector search is available, use it
        if self._vector_backend:
            try:
                if isinstance(self._vector_backend, QdrantMemoryBackend):
                    # Use vector search with filters in Qdrant
                    results = self._vector_backend.search(query, k=limit, since=since, min_importance=min_importance, type_filter=type_)
                else:
                    # Other backend (e.g., FAISS) - no built-in filtering support
                    results = self._vector_backend.search(query, k=limit)
                    # Post-filter the results by metadata if possible (not applicable for simple text lists, skip)
                # If we got enough results from vector search, return them
                if results:
                    return results[:limit]
            except Exception as e:
                logger.error(f"MemoryStore.search: Vector search failed (falling back to keyword search): {e}")
                results = []
        # Fallback to keyword-based search in SQLite if no vector results or vector backend is unavailable
        try:
            if self._fts_enabled:
                # Use full-text search (FTS5) on events content
                # Join with events table to retrieve metadata for filtering
                query_str = query.replace(",", " ")
                sql = ("SELECT e.content, e.ts, e.importance, e.type "
                       "FROM events_fts f JOIN events e ON f.rowid = e.id "
                       "WHERE f MATCH ? ORDER BY e.id DESC LIMIT ?")
                cur = self.conn.execute(sql, (query_str, limit * 3))  # fetch more than needed for filtering
            else:
                # FTS not available, use LIKE query
                sql = ("SELECT content, ts, importance, type FROM events "
                       "WHERE content LIKE ? ESCAPE '\\' "
                       "ORDER BY id DESC LIMIT ?")
                cur = self.conn.execute(sql, (f"%{query}%", limit * 3))
            rows = cur.fetchall()
        except Exception as e:
            logger.error(f"MemoryStore.search: Database keyword search query failed: {e}")
            return results  # return any results we might have from vector search (possibly empty)
        # Apply filtering conditions to the fetched rows
        for r in rows:
            content = r["content"]
            ts = r["ts"]
            imp = float(r["importance"] or 0.0)
            typ = r["type"]
            if since is not None:
                # Only include events on or after the 'since' timestamp
                # Compare using datetime for accuracy
                try:
                    ev_time = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    since_time = datetime.fromisoformat(since.replace("Z", "+00:00"))
                except Exception:
                    ev_time = None
                    since_time = None
                if ev_time and since_time:
                    if ev_time < since_time:
                        continue  # event is older than the since cutoff
                else:
                    # Fallback to simple string comparison if parsing fails
                    if ts < since:
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
            "created_at": datetime.now(timezone.utc).isoformat(),
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
            rows = self.list_events(type_="diagnostic", limit=limit)  # <-- your existing helper if present
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

    def keyword_search(self, query: str, limit: int = 5) -> list[dict]:
        """
        Search memory for a keyword or phrase.
        Tries FTS first, falls back to LIKE if needed.
        """
        try:
            cur = self.conn.execute(
                "SELECT * FROM memories WHERE content MATCH ? LIMIT ?", (query, limit)
            )
            return [dict(r) for r in cur.fetchall()]
        except Exception:
            cur = self.conn.execute(
                "SELECT * FROM memories WHERE content LIKE ? LIMIT ?", (f"%{query}%", limit)
            )
            return [dict(r) for r in cur.fetchall()]

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
