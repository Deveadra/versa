PRAGMA journal_mode=WAL;

-- Facts table with embeddings
CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key TEXT UNIQUE,
    value TEXT NOT NULL,
    last_updated TEXT NOT NULL,
    confidence REAL DEFAULT 0.75,
    last_reinforced DATETIME,
    embedding BLOB
);

-- Events + FTS
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content TEXT NOT NULL,
    ts TEXT NOT NULL,
    importance REAL NOT NULL DEFAULT 0,
    type TEXT NOT NULL DEFAULT 'event'
);
CREATE VIRTUAL TABLE IF NOT EXISTS events_fts
USING fts5(content, content='events', content_rowid='id');

-- Usage log
CREATE TABLE IF NOT EXISTS usage_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_text TEXT,
    normalized_intent TEXT,
    resolved_action TEXT,
    params_json TEXT,
    success INTEGER,
    latency_ms INTEGER,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Habits
CREATE TABLE IF NOT EXISTS habits (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key TEXT NOT NULL UNIQUE,
    count INTEGER NOT NULL DEFAULT 0,
    score REAL NOT NULL DEFAULT 0.0,
    last_used DATETIME
);

-- Feedback events
CREATE TABLE IF NOT EXISTS feedback_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    usage_id INTEGER,
    kind TEXT NOT NULL,
    note TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (usage_id) REFERENCES usage_log(id)
);

-- Policy assignments
CREATE TABLE IF NOT EXISTS policy_assignments (
    usage_id INTEGER PRIMARY KEY,
    policy_id TEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Context / signals
CREATE TABLE IF NOT EXISTS context_signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    value TEXT,
    type TEXT NOT NULL CHECK(type IN ('boolean','integer','float')) DEFAULT 'counter',
    description TEXT DEFAULT '',
    confidence REAL DEFAULT 1.0,
    source TEXT,
    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS derived_signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    definition TEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Engagement rules
CREATE TABLE IF NOT EXISTS engagement_rules (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    enabled INTEGER NOT NULL DEFAULT 1,
    topic_id TEXT NOT NULL,
    priority INTEGER NOT NULL DEFAULT 50,
    reset_signals TEXT DEFAULT NULL,
    cooldown_seconds INTEGER NOT NULL DEFAULT 1800,
    max_per_day INTEGER NOT NULL DEFAULT 8,
    condition_json TEXT NOT NULL,
    tone_strategy_json TEXT NOT NULL,
    context_template TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS rule_stats (
    rule_id INTEGER PRIMARY KEY,
    last_fired DATETIME,
    fires_today INTEGER NOT NULL DEFAULT 0,
    ema_success REAL NOT NULL DEFAULT 0.5,
    ema_negative REAL NOT NULL DEFAULT 0.5,
    FOREIGN KEY(rule_id) REFERENCES engagement_rules(id)
);

CREATE TABLE IF NOT EXISTS rule_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    rule_id INTEGER NOT NULL,
    fired_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    topic_id TEXT NOT NULL,
    tone TEXT,
    context TEXT,
    outcome TEXT,
    FOREIGN KEY(rule_id) REFERENCES engagement_rules(id)
);

CREATE TABLE IF NOT EXISTS rule_audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    rule_name TEXT NOT NULL,
    topic_id TEXT NOT NULL,
    rationale TEXT NOT NULL,
    details_json TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Tone memory
CREATE TABLE IF NOT EXISTS tone_memory (
    id INTEGER PRIMARY KEY,
    topic_id TEXT,
    tone TEXT,
    ignored_count INTEGER DEFAULT 0,
    acted_count INTEGER DEFAULT 0,
    consequence_note TEXT,
    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_tone TEXT,
    last_outcome TEXT,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Topics (canonical, single version)
CREATE TABLE IF NOT EXISTS topics (
    topic_id TEXT PRIMARY KEY,
    policy TEXT NOT NULL CHECK(policy IN ('principled','advocate','adaptive')),
    conviction REAL NOT NULL DEFAULT 0.75,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS topic_overrides (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic_id TEXT NOT NULL,
    type TEXT NOT NULL CHECK(type IN ('hard','soft','preference')),
    reason TEXT,
    expires_at DATETIME,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(topic_id) REFERENCES topics(topic_id)
);

CREATE TABLE IF NOT EXISTS topic_state (
    topic_id TEXT PRIMARY KEY,
    ignore_count INTEGER NOT NULL DEFAULT 0,
    escalation_count INTEGER NOT NULL DEFAULT 0,
    last_mentioned DATETIME
);

CREATE TABLE IF NOT EXISTS topic_feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic_id TEXT NOT NULL,
    feedback TEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(topic_id) REFERENCES topics(topic_id)
);

-- Consequence map
CREATE TABLE IF NOT EXISTS consequence_map (
    id INTEGER PRIMARY KEY,
    keyword TEXT UNIQUE,
    topic_id TEXT,
    confidence REAL DEFAULT 0.8,
    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
);
INSERT OR IGNORE INTO consequence_map (keyword, topic_id) VALUES
  ('headache', 'hydration'),
  ('fatigue', 'sleep'),
  ('tired', 'sleep'),
  ('leg', 'movement'),
  ('back', 'movement'),
  ('stress', 'workload'),
  ('late', 'time_management');

-- Complaint clusters
CREATE TABLE IF NOT EXISTS complaint_clusters (
    id INTEGER PRIMARY KEY,
    cluster TEXT,
    topic_id TEXT,
    examples TEXT,
    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_example TEXT DEFAULT NULL
);

-- Proposed rules
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
    rationale TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    approved_at TEXT,
    denied_at TEXT,
    applied_at TEXT,
    reverted_at TEXT
);
