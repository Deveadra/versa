# src/base/database/sqlite.py

from __future__ import annotations

import sqlite3
from pathlib import Path

from loguru import logger


class SQLiteConn:
    def __init__(self, path: str):
        self.path = Path(path)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

        self._configure_connection()
        self._init_db()

    def _configure_connection(self) -> None:
        """
        Connection-level safety defaults.
        These should be true for *every* DB connection Ultron uses.
        """
        cur = self.conn.cursor()

        # Enforce FK constraints (SQLite defaults OFF)
        cur.execute("PRAGMA foreign_keys = ON;")

        # Durability/perf defaults (safe for WAL)
        cur.execute("PRAGMA journal_mode = WAL;")
        cur.execute("PRAGMA synchronous = NORMAL;")
        cur.execute("PRAGMA temp_store = MEMORY;")

        # Avoid immediate 'database is locked' failures under light contention
        cur.execute("PRAGMA busy_timeout = 5000;")

        self.conn.commit()

    @staticmethod
    def _apply_fallback_schema(conn: sqlite3.Connection) -> None:
        """
        Fallback schema creation only when migrations are missing.
        Keep this *minimal* and aligned with the main schema where possible.
        """
        cur = conn.cursor()

        # Core tables (aligned with 0001_init.sql)
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT UNIQUE,
                value TEXT NOT NULL,
                last_updated TEXT NOT NULL,
                confidence REAL DEFAULT 0.75,
                last_reinforced DATETIME,
                embedding BLOB
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                ts TEXT NOT NULL,
                importance REAL NOT NULL DEFAULT 0,
                type TEXT NOT NULL DEFAULT 'event'
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS usage_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_text TEXT,
                normalized_intent TEXT,
                resolved_action TEXT,
                params_json TEXT,
                success INTEGER,
                latency_ms INTEGER,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS habits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT NOT NULL UNIQUE,
                count INTEGER NOT NULL DEFAULT 0,
                score REAL NOT NULL DEFAULT 0.0,
                last_used DATETIME
            )
            """
        )

        conn.commit()

    def _init_db(self) -> None:
        """
        Run tracked migrations if available, otherwise fallback schema.
        """
        logger.info("Initializing database / applying migrations if needed")

        mig_dir = Path(__file__).parent / "migrations"
        files = sorted(p for p in mig_dir.glob("*.sql"))

        # Track applied migrations
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL UNIQUE,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self.conn.commit()

        if not files:
            logger.warning("No migration files found; using fallback schema.")
            self._apply_fallback_schema(self.conn)
            return

        applied = {
            row["filename"]
            for row in self.conn.execute("SELECT filename FROM schema_migrations")
        }

        for p in files:
            if p.name in applied:
                continue

            sql = p.read_text(encoding="utf-8")
            try:
                logger.info(f"Applying migration {p.name}")
                self.conn.executescript(sql)
                self.conn.execute(
                    "INSERT INTO schema_migrations(filename) VALUES (?)",
                    (p.name,),
                )
                self.conn.commit()
            except Exception as e:
                self.conn.rollback()
                logger.exception(f"Failed to apply migration {p.name}: {e}")
                raise

    def cursor(self):
        return self.conn.cursor()

    def commit(self):
        return self.conn.commit()

    def execute(self, *args, **kwargs):
        return self.conn.execute(*args, **kwargs)

    def close(self):
        self.conn.close()
