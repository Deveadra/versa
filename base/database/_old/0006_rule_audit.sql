BEGIN;

CREATE TABLE IF NOT EXISTS rule_audit (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  rule_name TEXT NOT NULL,
  topic_id TEXT NOT NULL,
  rationale TEXT NOT NULL,     -- human-readable explanation
  details_json TEXT,           -- raw data (signals, feedback, stats)
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

COMMIT;
