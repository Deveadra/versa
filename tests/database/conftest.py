# tests/database/conftest.py
from __future__ import annotations


import pytest
import sqlite3

from pathlib import Path
from typing import Generator

from base.database.sqlite import SQLiteConn


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MIGRATIONS_DIR = PROJECT_ROOT / "src" / "base" / "database" / "migrations"
INIT_SQL = MIGRATIONS_DIR / "0001_init.sql"


def apply_sql(conn: sqlite3.Connection, sql_path: Path) -> None:
    sql = sql_path.read_text(encoding="utf-8")
    conn.executescript(sql)


@pytest.fixture()
def db(tmp_path) -> Generator[SQLiteConn, None, None]:
    db_path = tmp_path / "ultron_test.db"
    conn = SQLiteConn(str(db_path))
    try:
        yield conn
    finally:
        conn.close()
        

@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "test_ultron.db"


@pytest.fixture()
def raw_conn(db_path: Path) -> Generator[sqlite3.Connection, None, None]:
    conn = sqlite3.connect(db_path)
    try:
        # Safety defaults; your sqlite.py may override/expand these.
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA busy_timeout = 2000;")
        apply_sql(conn, INIT_SQL)
        yield conn
    finally:
        conn.close()
