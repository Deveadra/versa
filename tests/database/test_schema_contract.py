# tests/database/test_schema_contract.py
from __future__ import annotations

import sqlite3

from tests.database.schema_contract import REQUIRED_TABLES, REQUIRED_VIRTUAL_TABLES


def _tables(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
    ).fetchall()
    return {r[0] for r in rows}


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table});").fetchall()
    return {r[1] for r in rows}  # r[1] = column name


def test_required_tables_exist(db) -> None:
    existing = _tables(db.conn)
    missing = set(REQUIRED_TABLES.keys()) - existing
    assert not missing, f"Missing tables: {sorted(missing)}"


def test_required_columns_exist(db) -> None:
    for table, required_cols in REQUIRED_TABLES.items():
        cols = _columns(db.conn, table)
        missing = required_cols - cols
        assert not missing, f"{table} missing columns: {sorted(missing)}"


def test_virtual_tables_exist(db) -> None:
    existing = _tables(db.conn)
    missing = REQUIRED_VIRTUAL_TABLES - existing
    assert not missing, f"Missing virtual tables: {sorted(missing)}"
    