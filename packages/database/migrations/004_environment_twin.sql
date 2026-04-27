CREATE TABLE IF NOT EXISTS environments (
  id TEXT PRIMARY KEY,
  slug TEXT NOT NULL UNIQUE,
  name TEXT NOT NULL,
  metadata_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  last_validated_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_environments_updated ON environments(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_environments_last_validated ON environments(last_validated_at DESC);

CREATE TABLE IF NOT EXISTS environment_records (
  id TEXT PRIMARY KEY,
  environment_id TEXT NOT NULL,
  kind TEXT NOT NULL,
  name TEXT NOT NULL,
  description TEXT,
  attributes_json TEXT NOT NULL,
  metadata_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  validated_at TEXT,
  FOREIGN KEY(environment_id) REFERENCES environments(id)
);

CREATE INDEX IF NOT EXISTS idx_environment_records_environment_updated
  ON environment_records(environment_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS environment_relationships (
  id TEXT PRIMARY KEY,
  environment_id TEXT NOT NULL,
  from_entity_id TEXT NOT NULL,
  to_entity_id TEXT NOT NULL,
  relation TEXT NOT NULL,
  direction TEXT NOT NULL,
  notes TEXT,
  created_at TEXT NOT NULL,
  FOREIGN KEY(environment_id) REFERENCES environments(id)
);

CREATE INDEX IF NOT EXISTS idx_environment_relationships_environment_created
  ON environment_relationships(environment_id, created_at DESC);

CREATE TABLE IF NOT EXISTS environment_access_paths (
  id TEXT PRIMARY KEY,
  environment_id TEXT NOT NULL,
  entity_id TEXT NOT NULL,
  name TEXT NOT NULL,
  method TEXT NOT NULL,
  endpoint TEXT NOT NULL,
  prerequisites_json TEXT NOT NULL,
  command_ref_ids_json TEXT NOT NULL,
  notes TEXT,
  created_at TEXT NOT NULL,
  validated_at TEXT,
  FOREIGN KEY(environment_id) REFERENCES environments(id)
);

CREATE INDEX IF NOT EXISTS idx_environment_access_paths_environment_created
  ON environment_access_paths(environment_id, created_at DESC);

CREATE TABLE IF NOT EXISTS environment_procedures (
  id TEXT PRIMARY KEY,
  environment_id TEXT NOT NULL,
  name TEXT NOT NULL,
  intent TEXT NOT NULL,
  target_entity_ids_json TEXT NOT NULL,
  steps_json TEXT NOT NULL,
  last_validated_at TEXT,
  owner TEXT,
  tags_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  FOREIGN KEY(environment_id) REFERENCES environments(id)
);

CREATE INDEX IF NOT EXISTS idx_environment_procedures_environment_updated
  ON environment_procedures(environment_id, updated_at DESC);
