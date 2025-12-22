# tests/database/test_migrations.py
from __future__ import annotations

import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INIT_SQL = PROJECT_ROOT / "src" / "base" / "database" / "migrations" / "0001_init.sql"


def test_init_migration_applies_cleanly(tmp_path: Path) -> None:
    db_file = tmp_path / "migrate.db"
    conn = sqlite3.connect(db_file)
    try:
        sql = INIT_SQL.read_text(encoding="utf-8")
        conn.executescript(sql)
    finally:
        conn.close()


def test_init_migration_is_idempotent_enough(tmp_path: Path) -> None:
    """
    If your migration uses CREATE TABLE IF NOT EXISTS, this should pass.
    If not, we’ll adjust policy:
      - either enforce idempotent migrations
      - or create a tiny migrations table + runner that tracks applied versions
    """
    db_file = tmp_path / "migrate_twice.db"
    conn = sqlite3.connect(db_file)
    try:
        sql = INIT_SQL.read_text(encoding="utf-8")
        conn.executescript(sql)
        conn.executescript(sql)  # apply twice
    finally:
        conn.close()
