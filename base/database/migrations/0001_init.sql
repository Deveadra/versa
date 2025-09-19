
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS facts (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    last_updated TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content TEXT NOT NULL,
    ts TEXT NOT NULL,
    importance REAL NOT NULL DEFAULT 0,
    type TEXT NOT NULL DEFAULT 'event'
);

-- Optional: simple FTS for keyword fallback
CREATE VIRTUAL TABLE IF NOT EXISTS events_fts USING fts5(content, content='events', content_rowid='id');