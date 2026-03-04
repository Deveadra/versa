-- Repo Janitor / Self-improve flywheel tables (authoritative)

CREATE TABLE IF NOT EXISTS repo_score_runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
  run_type TEXT NOT NULL,
  mode TEXT NOT NULL,
  fix_enabled INTEGER NOT NULL DEFAULT 0,
  git_branch TEXT,
  git_sha TEXT,
  score REAL NOT NULL DEFAULT 0,
  passed INTEGER NOT NULL DEFAULT 0,
  metrics_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_repo_score_runs_created_at ON repo_score_runs(created_at);
CREATE INDEX IF NOT EXISTS idx_repo_score_runs_run_type ON repo_score_runs(run_type);

CREATE TABLE IF NOT EXISTS repo_improvement_attempts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
  iteration INTEGER NOT NULL,
  baseline_run_id INTEGER NOT NULL,
  before_run_id INTEGER NOT NULL,
  after_run_id INTEGER,
  branch TEXT NOT NULL,
  proposal_title TEXT,
  proposal_json TEXT,
  pr_url TEXT,
  improved INTEGER NOT NULL DEFAULT 0,
  error_text TEXT,

  FOREIGN KEY (baseline_run_id) REFERENCES repo_score_runs(id),
  FOREIGN KEY (before_run_id) REFERENCES repo_score_runs(id),
  FOREIGN KEY (after_run_id) REFERENCES repo_score_runs(id)
);

CREATE INDEX IF NOT EXISTS idx_repo_improvement_attempts_created_at ON repo_improvement_attempts(created_at);
CREATE INDEX IF NOT EXISTS idx_repo_improvement_attempts_branch ON repo_improvement_attempts(branch);

CREATE TABLE IF NOT EXISTS capability_gaps (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
  source TEXT NOT NULL,
  fingerprint TEXT NOT NULL UNIQUE,
  requested_capability TEXT NOT NULL,
  observed_failure TEXT,
  classification TEXT NOT NULL,
  repro_steps TEXT,
  priority INTEGER NOT NULL DEFAULT 0,
  status TEXT NOT NULL DEFAULT 'queued',
  metadata_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_capability_gaps_status_priority ON capability_gaps(status, priority);
CREATE INDEX IF NOT EXISTS idx_capability_gaps_created_at ON capability_gaps(created_at);
CREATE INDEX IF NOT EXISTS idx_capability_gaps_status ON capability_gaps(status);