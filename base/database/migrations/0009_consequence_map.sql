BEGIN;

CREATE TABLE IF NOT EXISTS consequence_map (
  id INTEGER PRIMARY KEY,
  keyword TEXT UNIQUE,
  topic_id TEXT,
  confidence REAL DEFAULT 0.8,
  last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Seed values
INSERT OR IGNORE INTO consequence_map (keyword, topic_id) VALUES
  ('headache', 'hydration'),
  ('fatigue', 'sleep'),
  ('tired', 'sleep'),
  ('leg', 'movement'),
  ('back', 'movement'),
  ('stress', 'workload'),
  ('late', 'time_management');

CREATE TABLE IF NOT EXISTS complaint_clusters (
  id INTEGER PRIMARY KEY,
  cluster TEXT,
  topic_id TEXT,
  examples TEXT,  -- JSON array of raw complaints
  last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
  last_example TEXT DEFAULT NULL;
);

COMMIT;
