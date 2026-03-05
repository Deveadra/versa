from __future__ import annotations

import sqlite3
from collections.abc import Generator
from pathlib import Path

import pytest

from base.database.sqlite import SQLiteConn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MIGRATIONS_DIR = PROJECT_ROOT / "src" / "base" / "database" / "migrations"
INIT_SQL_FILES = sorted(MIGRATIONS_DIR.glob("*.sql"))


def apply_sql(conn: sqlite3.Connection, sql_path: Path) -> None:
    sql = sql_path.read_text(encoding="utf-8")
    conn.executescript(sql)


@pytest.fixture()
def db(tmp_path) -> Generator[SQLiteConn, None, None]:
    db_path = tmp_path / "aerith_test.db"
    conn = SQLiteConn(str(db_path))
    try:
        yield conn
    finally:
        conn.close()


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "test_aerith.db"


@pytest.fixture()
def raw_conn(db_path: Path) -> Generator[sqlite3.Connection, None, None]:
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA busy_timeout = 2000;")
        for p in INIT_SQL_FILES:
            apply_sql(conn, p)
        yield conn
    finally:
        conn.close()
