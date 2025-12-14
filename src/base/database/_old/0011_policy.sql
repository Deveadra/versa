BEGIN;

-- Topics Ultron may proactively speak about
CREATE TABLE IF NOT EXISTS topics (
  topic_id TEXT PRIMARY KEY,
  policy   TEXT NOT NULL CHECK(policy IN ('principled','advocate','adaptive')),
  conviction REAL NOT NULL DEFAULT 0.75,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- User overrides: hard (never), soft (pause until expires), preference (tone/channel tweaks)
CREATE TABLE IF NOT EXISTS topic_overrides (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  topic_id TEXT NOT NULL,
  type TEXT NOT NULL CHECK(type IN ('hard','soft','preference')),
  reason TEXT,
  expires_at DATETIME,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY(topic_id) REFERENCES topics(topic_id)
);

-- Live state for scoring/pacing
CREATE TABLE IF NOT EXISTS topic_state (
  topic_id TEXT PRIMARY KEY,
  ignore_count INTEGER NOT NULL DEFAULT 0,
  escalation_count INTEGER NOT NULL DEFAULT 0,
  last_mentioned DATETIME
);

-- Feedback signals to tune tone/pace
CREATE TABLE IF NOT EXISTS topic_feedback (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  topic_id TEXT NOT NULL,
  feedback TEXT NOT NULL, -- 'acted','thanks','ignore','angry'
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY(topic_id) REFERENCES topics(topic_id)
);

COMMIT;
