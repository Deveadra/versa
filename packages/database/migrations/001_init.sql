CREATE TABLE IF NOT EXISTS tasks (
  id TEXT PRIMARY KEY,
  title TEXT NOT NULL,
  description TEXT,
  status TEXT NOT NULL,
  priority TEXT NOT NULL,
  due_date TEXT,
  scheduled_date TEXT,
  tags_json TEXT NOT NULL DEFAULT '[]',
  linked_goal_id TEXT,
  domain TEXT,
  source TEXT NOT NULL DEFAULT 'manual',
  completed_at TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tasks_status_due ON tasks(status, due_date);
CREATE INDEX IF NOT EXISTS idx_tasks_goal ON tasks(linked_goal_id);

CREATE TABLE IF NOT EXISTS goals (
  id TEXT PRIMARY KEY,
  title TEXT NOT NULL,
  description TEXT,
  domain TEXT NOT NULL DEFAULT 'core',
  target_type TEXT,
  target_value REAL,
  current_value REAL,
  deadline TEXT,
  status TEXT NOT NULL,
  why_it_matters TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_goals_status_deadline ON goals(status, deadline);

CREATE TABLE IF NOT EXISTS schedule_blocks (
  id TEXT PRIMARY KEY,
  title TEXT NOT NULL,
  type TEXT NOT NULL DEFAULT 'focus',
  start_time TEXT NOT NULL,
  end_time TEXT NOT NULL,
  date TEXT NOT NULL,
  linked_task_id TEXT,
  linked_goal_id TEXT,
  domain TEXT,
  notes TEXT,
  status TEXT NOT NULL DEFAULT 'scheduled',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_schedule_day ON schedule_blocks(date, start_time);

CREATE TABLE IF NOT EXISTS study_courses (
  id TEXT PRIMARY KEY,
  title TEXT NOT NULL,
  code TEXT,
  term TEXT,
  instructor TEXT,
  status TEXT NOT NULL DEFAULT 'active',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS study_assignments (
  id TEXT PRIMARY KEY,
  course_id TEXT NOT NULL,
  title TEXT NOT NULL,
  type TEXT,
  due_date TEXT,
  status TEXT NOT NULL DEFAULT 'todo',
  notes TEXT,
  estimated_effort INTEGER,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_study_assignments_due ON study_assignments(status, due_date);

CREATE TABLE IF NOT EXISTS study_sessions (
  id TEXT PRIMARY KEY,
  course_id TEXT NOT NULL,
  assignment_id TEXT,
  start_time TEXT NOT NULL,
  end_time TEXT,
  planned_duration INTEGER,
  actual_duration INTEGER,
  method TEXT,
  notes TEXT,
  outcome TEXT,
  focus_score INTEGER,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS job_leads (
  id TEXT PRIMARY KEY,
  company TEXT NOT NULL,
  role TEXT NOT NULL,
  source TEXT,
  link TEXT,
  location TEXT,
  compensation TEXT,
  status TEXT NOT NULL DEFAULT 'lead',
  notes TEXT,
  discovered_at TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS job_applications (
  id TEXT PRIMARY KEY,
  lead_id TEXT,
  resume_asset_id TEXT,
  cover_letter_ref TEXT,
  status TEXT NOT NULL,
  applied_at TEXT,
  follow_up_date TEXT,
  notes TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_job_apps_followup ON job_applications(status, follow_up_date);

CREATE TABLE IF NOT EXISTS resume_assets (
  id TEXT PRIMARY KEY,
  label TEXT NOT NULL,
  storage_ref TEXT,
  version TEXT,
  target_role TEXT,
  notes TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS system_events (
  event_id TEXT PRIMARY KEY,
  event_type TEXT NOT NULL,
  actor TEXT NOT NULL,
  timestamp TEXT NOT NULL,
  domain TEXT NOT NULL,
  entity_type TEXT NOT NULL,
  entity_id TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  sensitivity TEXT NOT NULL,
  trace_id TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON system_events(timestamp DESC);

CREATE TABLE IF NOT EXISTS consent_grants (
  id TEXT PRIMARY KEY,
  scope TEXT NOT NULL,
  granted INTEGER NOT NULL,
  granted_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS integration_accounts (
  id TEXT PRIMARY KEY,
  provider TEXT NOT NULL,
  account_label TEXT NOT NULL,
  connected_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS memories (
  id TEXT PRIMARY KEY,
  tier TEXT NOT NULL,
  summary TEXT NOT NULL,
  content_json TEXT NOT NULL,
  metadata_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  last_accessed_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_memories_tier_updated ON memories(tier, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_memories_updated ON memories(updated_at DESC);

CREATE TABLE IF NOT EXISTS migrations (
  id TEXT PRIMARY KEY,
  applied_at TEXT NOT NULL
);
