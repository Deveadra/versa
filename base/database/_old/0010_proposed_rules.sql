BEGIN;

CREATE TABLE IF NOT EXISTS proposed_rules (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    topic_id TEXT,
    priority INTEGER,
    cooldown_seconds INTEGER,
    max_per_day INTEGER,
    condition_json TEXT,
    tone_strategy_json TEXT,
    context_template TEXT,
    confidence REAL,
    score REAL,
    status TEXT DEFAULT 'pending',
    approved_at TEXT,
    denied_at TEXT,
    applied_at TEXT,
    reverted_at TEXT;
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    rationale TEXT,
    status TEXT DEFAULT 'pending'  -- 'pending' | 'accepted' | 'rejected' | 'reverted'
);

COMMIT;