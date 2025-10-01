BEGIN;

CREATE TABLE IF NOT EXISTS tone_memory (
  id INTEGER PRIMARY KEY,
  topic_id TEXT,
  tone TEXT,
  ignored_count INTEGER DEFAULT 0,
  acted_count INTEGER DEFAULT 0,
  consequence_note TEXT,
  last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
  last_tone TEXT,
  last_outcome TEXT,          -- "acted", "thanks", "ignore", "angry"
  updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);


COMMIT;
