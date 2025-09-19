
BEGIN;

-- 1) Usage log for *everything* Ultron does or is asked to do
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

-- 2) Habits table: recency‑weighted counts + last_used
CREATE TABLE IF NOT EXISTS habits (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  key TEXT NOT NULL,
  count INTEGER NOT NULL DEFAULT 0,
  score REAL NOT NULL DEFAULT 0.0,
  last_used DATETIME,
  UNIQUE(key)
);

-- 3) Feedback confirmations/corrections
CREATE TABLE IF NOT EXISTS feedback_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  usage_id INTEGER,
  kind TEXT NOT NULL,
  note TEXT,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (usage_id) REFERENCES usage_log(id)
);

-- 4) Facts table columns for confidence maintenance (if not present)
-- (SQLite will error if column exists; wrap in try via separate statements)
-- We'll attempt; if it fails in your environment, keep the existing column.
-- Add confidence
ALTER TABLE facts ADD COLUMN confidence REAL DEFAULT 0.75;
-- Add last_reinforced
ALTER TABLE facts ADD COLUMN last_reinforced DATETIME;

COMMIT;
