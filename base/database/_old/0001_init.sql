
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS facts (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    last_updated TEXT NOT NULL,
    confidence REAL DEFAULT 0.75,
    last_reinforced DATETIME
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content TEXT NOT NULL,
    ts TEXT NOT NULL,
    importance REAL NOT NULL DEFAULT 0,
    type TEXT NOT NULL DEFAULT 'event'
);

-- Extend facts table for embeddings
-- When creating the table in sqlite.py (or wherever your DB schema lives), add an embedding column.

CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key TEXT,
    value TEXT,
    last_updated TEXT,
    embedding BLOB
);

-- Optional: simple FTS for keyword fallback
CREATE VIRTUAL TABLE IF NOT EXISTS events_fts USING fts5(content, content='events', content_rowid='id');