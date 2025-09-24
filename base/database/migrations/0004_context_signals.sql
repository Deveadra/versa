BEGIN;

CREATE TABLE IF NOT EXISTS context_signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    value TEXT, -- boolean/int/float stored as string
    confidence REAL DEFAULT 1.0,
    source TEXT,
    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS derived_signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    definition TEXT NOT NULL, -- JSON conditions, e.g. {"all":[{"signal":"hydration_gap","value":true},{"signal":"time_of_day","gte":"21:00"}]}
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

COMMIT;
