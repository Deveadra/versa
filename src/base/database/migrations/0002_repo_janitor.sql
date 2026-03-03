-- Repo Janitor: scoreboard runs, attempts, and capability gaps

CREATE TABLE IF NOT EXISTS repo_score_runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  run_type TEXT NOT NULL,              -- baseline|iteration_before|iteration_after
  mode TEXT NOT NULL,                  -- all|changed
  fix_enabled INTEGER NOT NULL DEFAULT 0,
  git_branch TEXT,
  git_sha TEXT,
  score REAL NOT NULL,
  passed INTEGER NOT NULL DEFAULT 0,
  metrics_json TEXT NOT NULL           -- json blob
);

CREATE TABLE IF NOT EXISTS repo_improvement_attempts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  iteration INTEGER NOT NULL,
  baseline_run_id INTEGER NOT NULL,
  after_run_id INTEGER NOT NULL,
  proposal_title TEXT,
  proposal_json TEXT,
  branch TEXT,
  pr_url TEXT,
  patch_summary_json TEXT,
  improved INTEGER NOT NULL DEFAULT 0,
  regression_json TEXT,
  error_text TEXT,
  FOREIGN KEY(baseline_run_id) REFERENCES repo_score_runs(id),
  FOREIGN KEY(after_run_id) REFERENCES repo_score_runs(id)
);

CREATE TABLE IF NOT EXISTS capability_gaps (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  gap_type TEXT NOT NULL,              -- missing_tool|formatting|lint|tests|syntax|blocked|bug|unknown
  component TEXT,
  signature TEXT NOT NULL UNIQUE,      -- stable key for upsert
  context_json TEXT,
  status TEXT NOT NULL DEFAULT 'open',
  last_seen_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  seen_count INTEGER NOT NULL DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_repo_score_runs_created_at ON repo_score_runs(created_at);
CREATE INDEX IF NOT EXISTS idx_repo_attempts_created_at ON repo_improvement_attempts(created_at);
CREATE INDEX IF NOT EXISTS idx_capability_gaps_status ON capability_gaps(status);
