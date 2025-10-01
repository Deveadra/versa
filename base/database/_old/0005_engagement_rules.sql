BEGIN;

CREATE TABLE IF NOT EXISTS engagement_rules (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL UNIQUE,
  enabled INTEGER NOT NULL DEFAULT 1,
  topic_id TEXT NOT NULL,           -- e.g. "hydration", "posture", "screen_time"
  priority INTEGER NOT NULL DEFAULT 50,
  reset_signals TEXT DEFAULT NULL,  -- comma-separated signal names to reset on "acted" outcome
  cooldown_seconds INTEGER NOT NULL DEFAULT 1800,   -- min gap per rule
  max_per_day INTEGER NOT NULL DEFAULT 8,

  -- JSON: JSON-Logic-like condition object evaluated against context signals.
  -- Example: {"all":[{"eq":["signal:sedentary_minutes>=120", true]},{"eq":["signal:work_session_active", true]}]}
  condition_json TEXT NOT NULL,

  -- JSON: determines tone from computed "severity" or context.
  -- Example: {"map":[{"gte":[ "severity", 0.7], "tone":"firm"},
  --                  {"gte":[ "severity", 0.4], "tone":"persistent"}],
  --           "default":"gentle"}
  tone_strategy_json TEXT NOT NULL,

  -- Template with placeholders bound from signals. Example: "sitting_minutes={{sedentary_minutes}}"
  context_template TEXT,

  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS rule_stats (
  rule_id INTEGER PRIMARY KEY,
  last_fired DATETIME,
  fires_today INTEGER NOT NULL DEFAULT 0,
  ema_success REAL NOT NULL DEFAULT 0.5,   -- exponential moving avg of positive outcomes
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
  outcome TEXT,            -- "acted","thanks","ignore","angry"
  FOREIGN KEY(rule_id) REFERENCES engagement_rules(id)
);

COMMIT;
