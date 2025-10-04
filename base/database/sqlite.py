from __future__ import annotations

import sqlite3
from pathlib import Path

from loguru import logger



class SQLiteConn:
    def __init__(self, path: str):
        self.path = Path(path)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_db()

    @staticmethod
    def init_db(path: str):
        """Fallback inline schema creation if migrations are missing."""
        conn = sqlite3.connect(path, check_same_thread=False)
        cur = conn.cursor()

        # Enable WAL
        cur.execute("PRAGMA journal_mode=WAL;")
        cur.execute("PRAGMA synchronous=NORMAL;")
        cur.execute("PRAGMA temp_store=MEMORY;")

        # Schema
        # PRAGMA journal_mode=WAL;
        # PRAGMA synchronous=NORMAL;
        # PRAGMA temp_store=MEMORY;

        # Events / Memories
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
        cur.execute(
            """
        CREATE TABLE IF NOT EXISTS facts (
            key TEXT PRIMARY KEY,
            value TEXT,
            last_updated TIMESTAMP
        )
        """
        )
        cur.execute(
            """
        CREATE TABLE IF NOT EXISTS habits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key TEXT,
            count INTEGER,
            score REAL,
            last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        )
        cur.execute(
            """
        CREATE TABLE IF NOT EXISTS usage_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            resolved_action TEXT,
            params_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        )

        conn.commit()
        return conn

    def _init_db(self):
        """Run migrations if available, otherwise fallback inline schema."""
        logger.info("Running migrations if needed")
        mig_dir = Path(__file__).parent / "migrations"
        files = sorted(p for p in mig_dir.glob("*.sql"))

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

        # --- get already applied migrations ---
        applied = {
            row["filename"] for row in self.conn.execute("SELECT filename FROM schema_migrations")
        }

        # --- apply migrations in order ---
        for p in files:
            if p.name not in applied:
                sql = p.read_text(encoding="utf-8")
                try:
                    logger.info(f"Applying migration {p.name}")
                    self.conn.executescript(sql)
                    self.conn.execute(
                        "INSERT INTO schema_migrations(filename) VALUES (?)", (p.name,)
                    )
                    self.conn.commit()
                except Exception as e:
                    logger.exception(f"Failed to apply migration {p.name}: {e}")
                    raise

        # Fallback if no migration files
        SQLiteConn.init_db(self.path)  # type: ignore

    def cursor(self):
        return self.conn.cursor()

    def commit(self):
        return self.conn.commit()

    def execute(self, *args, **kwargs):
        return self.conn.execute(*args, **kwargs)

    def close(self):
        self.conn.close()
