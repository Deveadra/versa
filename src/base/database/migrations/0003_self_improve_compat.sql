-- 0003_self_improve_compat.sql
-- Compatibility tables expected by older/self-improve migration tests.

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS dream_runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  goal TEXT,
  summary TEXT,
  status TEXT DEFAULT 'complete',
  metadata_json TEXT
);

CREATE TABLE IF NOT EXISTS dream_proposals (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id INTEGER,
  title TEXT,
  description TEXT,
  proposal_json TEXT,
  pr_url TEXT,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (run_id) REFERENCES dream_runs(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS dream_artifacts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id INTEGER,
  kind TEXT,
  path TEXT,
  payload_json TEXT,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (run_id) REFERENCES dream_runs(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS scoreboard (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_type TEXT NOT NULL,
  mode TEXT NOT NULL,
  fix_enabled INTEGER NOT NULL DEFAULT 0,
  git_branch TEXT,
  git_sha TEXT,
  score REAL NOT NULL DEFAULT 0.0,
  gates_failing INTEGER NOT NULL DEFAULT 0,
  passed INTEGER NOT NULL DEFAULT 0,
  metrics_json TEXT,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_scoreboard_created_at ON scoreboard(created_at);
CREATE INDEX IF NOT EXISTS idx_dream_runs_created_at ON dream_runs(created_at);
