CREATE TABLE IF NOT EXISTS workspaces (
  id TEXT PRIMARY KEY,
  slug TEXT NOT NULL UNIQUE,
  name TEXT NOT NULL,
  repository TEXT,
  metadata_json TEXT NOT NULL,
  state_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  last_activated_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_workspaces_updated ON workspaces(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_workspaces_last_activated ON workspaces(last_activated_at DESC);

CREATE TABLE IF NOT EXISTS workspace_checkpoints (
  id TEXT PRIMARY KEY,
  workspace_id TEXT NOT NULL,
  summary TEXT NOT NULL,
  snapshot_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  created_by TEXT NOT NULL,
  FOREIGN KEY(workspace_id) REFERENCES workspaces(id)
);

CREATE INDEX IF NOT EXISTS idx_workspace_checkpoints_workspace_created
  ON workspace_checkpoints(workspace_id, created_at DESC);
