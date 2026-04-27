import Database from 'better-sqlite3';
import { randomUUID } from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  deriveWorkspaceSummary,
  DomainEventSchema,
  EnvironmentAccessPathSchema,
  EnvironmentProcedureSchema,
  EnvironmentRecordSchema,
  EnvironmentRelationshipSchema,
  EnvironmentSummarySchema,
  EnvironmentTwinCreateRequestSchema,
  EnvironmentTwinRecordSchema,
  MemoryConsolidationRequestSchema,
  MemoryReadRequestSchema,
  MemoryRecordSchema,
  MemoryWriteRequestSchema,
  WorkspaceCheckpointCreateRequestSchema,
  WorkspaceCheckpointSchema,
  WorkspaceCreateRequestSchema,
  WorkspaceRecordSchema,
  WorkspaceStatePatchSchema,
  WorkspaceSummarySchema,
  normalizeEnvironmentItemLimit,
  normalizeWorkspaceCheckpointLimit,
  type EnvironmentAccessPath,
  type EnvironmentProcedure,
  type EnvironmentRecord,
  type EnvironmentRelationship,
  type EnvironmentSummary,
  type EnvironmentTwinCreateRequest,
  type EnvironmentTwinRecord,
  type MemoryConsolidationRequest,
  type MemoryReadRequest,
  type MemoryRecord,
  type MemoryWriteRequest,
  type WorkspaceCheckpoint,
  type WorkspaceCheckpointCreateRequest,
  type WorkspaceCreateRequest,
  type WorkspaceRecord,
  type WorkspaceStatePatch,
  type WorkspaceSummary,
} from '@versa/shared';

const defaultDbPath = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  '../data/versa.db',
);
const id = (prefix: string) => `${prefix}_${randomUUID().slice(0, 8)}`;
const now = () => new Date().toISOString();

export type TaskRecord = {
  id: string;
  title: string;
  description?: string;
  status: string;
  priority: string;
  created_at: string;
  updated_at: string;
};

export const connectDb = (filename = process.env.DATABASE_URL ?? defaultDbPath) => {
  fs.mkdirSync(path.dirname(filename), { recursive: true });
  return new Database(filename);
};

export const taskRepo = (db: Database.Database) => ({
  create: (input: {
    title: string;
    description?: string;
    priority?: string;
    dueDate?: string;
    scheduledDate?: string;
    tags?: string[];
    linkedGoalId?: string;
    domain?: string;
  }) => {
    const ts = now();
    const row = {
      id: id('tsk'),
      title: input.title,
      description: input.description ?? null,
      status: 'todo',
      priority: input.priority ?? 'medium',
      due_date: input.dueDate ?? null,
      scheduled_date: input.scheduledDate ?? null,
      tags_json: JSON.stringify(input.tags ?? []),
      linked_goal_id: input.linkedGoalId ?? null,
      domain: input.domain ?? 'core',
      source: 'manual',
      completed_at: null,
      created_at: ts,
      updated_at: ts,
    };
    db.prepare(
      `INSERT INTO tasks (id,title,description,status,priority,due_date,scheduled_date,tags_json,linked_goal_id,domain,source,completed_at,created_at,updated_at)
       VALUES (@id,@title,@description,@status,@priority,@due_date,@scheduled_date,@tags_json,@linked_goal_id,@domain,@source,@completed_at,@created_at,@updated_at)`,
    ).run(row);
    return row;
  },
  update: (taskId: string, patch: Record<string, unknown>) => {
    const existing = db.prepare('SELECT * FROM tasks WHERE id = ?').get(taskId) as
      | Record<string, unknown>
      | undefined;
    if (!existing) return null;
    const merged = {
      ...existing,
      ...patch,
      tags_json: patch.tags ? JSON.stringify(patch.tags) : existing.tags_json,
      updated_at: now(),
    };
    db.prepare(
      `UPDATE tasks SET title=@title, description=@description, status=@status, priority=@priority, due_date=@due_date,
      scheduled_date=@scheduled_date, tags_json=@tags_json, linked_goal_id=@linked_goal_id, domain=@domain,
      source=@source, completed_at=@completed_at, updated_at=@updated_at WHERE id=@id`,
    ).run(merged);
    return db.prepare('SELECT * FROM tasks WHERE id = ?').get(taskId);
  },
  list: (scope: 'all' | 'today' | 'overdue' = 'all') => {
    if (scope === 'today') {
      return db
        .prepare(
          "SELECT * FROM tasks WHERE status != 'done' AND date(COALESCE(scheduled_date,due_date)) = date('now') ORDER BY due_date",
        )
        .all();
    }
    if (scope === 'overdue') {
      return db
        .prepare(
          "SELECT * FROM tasks WHERE status != 'done' AND due_date IS NOT NULL AND datetime(due_date) < datetime('now') ORDER BY due_date",
        )
        .all();
    }
    return db.prepare('SELECT * FROM tasks ORDER BY created_at DESC').all();
  },
  remove: (taskId: string) =>
    db.prepare("UPDATE tasks SET status='archived', updated_at=? WHERE id=?").run(now(), taskId),
});

export const goalRepo = (db: Database.Database) => ({
  create: (input: {
    title: string;
    description?: string;
    domain?: string;
    deadline?: string;
    whyItMatters?: string;
  }) => {
    const ts = now();
    const row = {
      id: id('gol'),
      title: input.title,
      description: input.description ?? null,
      domain: input.domain ?? 'core',
      target_type: null,
      target_value: null,
      current_value: null,
      deadline: input.deadline ?? null,
      status: 'active',
      why_it_matters: input.whyItMatters ?? null,
      created_at: ts,
      updated_at: ts,
    };
    db.prepare(
      `INSERT INTO goals (id,title,description,domain,target_type,target_value,current_value,deadline,status,why_it_matters,created_at,updated_at)
       VALUES (@id,@title,@description,@domain,@target_type,@target_value,@current_value,@deadline,@status,@why_it_matters,@created_at,@updated_at)`,
    ).run(row);
    return row;
  },
  update: (goalId: string, patch: Record<string, unknown>) => {
    const existing = db.prepare('SELECT * FROM goals WHERE id=?').get(goalId) as
      | Record<string, unknown>
      | undefined;
    if (!existing) return null;
    const merged = { ...existing, ...patch, updated_at: now() };
    db.prepare(
      `UPDATE goals SET title=@title, description=@description, domain=@domain, target_type=@target_type, target_value=@target_value,
      current_value=@current_value, deadline=@deadline, status=@status, why_it_matters=@why_it_matters, updated_at=@updated_at WHERE id=@id`,
    ).run(merged);
    return db.prepare('SELECT * FROM goals WHERE id=?').get(goalId);
  },
  list: () => db.prepare('SELECT * FROM goals ORDER BY updated_at DESC').all(),
});

export const scheduleRepo = (db: Database.Database) => ({
  create: (input: {
    title: string;
    date: string;
    startTime: string;
    endTime: string;
    linkedTaskId?: string;
    linkedGoalId?: string;
  }) => {
    if (input.endTime <= input.startTime) {
      throw new Error('invalid time range');
    }
    const overlap = db
      .prepare(
        `SELECT id FROM schedule_blocks WHERE date = ? AND status != 'deleted'
         AND NOT (end_time <= ? OR start_time >= ?)`,
      )
      .get(input.date, input.startTime, input.endTime);
    if (overlap) {
      throw new Error('overlap detected');
    }
    const ts = now();
    const row = {
      id: id('blk'),
      title: input.title,
      type: 'focus',
      start_time: input.startTime,
      end_time: input.endTime,
      date: input.date,
      linked_task_id: input.linkedTaskId ?? null,
      linked_goal_id: input.linkedGoalId ?? null,
      domain: 'core',
      notes: null,
      status: 'scheduled',
      created_at: ts,
      updated_at: ts,
    };
    db.prepare(
      `INSERT INTO schedule_blocks (id,title,type,start_time,end_time,date,linked_task_id,linked_goal_id,domain,notes,status,created_at,updated_at)
       VALUES (@id,@title,@type,@start_time,@end_time,@date,@linked_task_id,@linked_goal_id,@domain,@notes,@status,@created_at,@updated_at)`,
    ).run(row);
    return row;
  },
  setStatus: (idValue: string, status: string) =>
    db
      .prepare('UPDATE schedule_blocks SET status=?, updated_at=? WHERE id=?')
      .run(status, now(), idValue),
  listDay: (date: string) =>
    db.prepare('SELECT * FROM schedule_blocks WHERE date=? ORDER BY start_time').all(date),
  listWeek: () =>
    db
      .prepare(
        "SELECT * FROM schedule_blocks WHERE date BETWEEN date('now') AND date('now','+7 day') ORDER BY date, start_time",
      )
      .all(),
});

export const studyRepo = (db: Database.Database) => ({
  createCourse: (title: string) => {
    const ts = now();
    const row = {
      id: id('crs'),
      title,
      code: null,
      term: null,
      instructor: null,
      status: 'active',
      created_at: ts,
      updated_at: ts,
    };
    db.prepare(
      'INSERT INTO study_courses (id,title,code,term,instructor,status,created_at,updated_at) VALUES (@id,@title,@code,@term,@instructor,@status,@created_at,@updated_at)',
    ).run(row);
    return row;
  },
  createAssignment: (courseId: string, title: string, dueDate?: string) => {
    const ts = now();
    const row = {
      id: id('asg'),
      course_id: courseId,
      title,
      type: 'homework',
      due_date: dueDate ?? null,
      status: 'todo',
      notes: null,
      estimated_effort: null,
      created_at: ts,
      updated_at: ts,
    };
    db.prepare(
      'INSERT INTO study_assignments (id,course_id,title,type,due_date,status,notes,estimated_effort,created_at,updated_at) VALUES (@id,@course_id,@title,@type,@due_date,@status,@notes,@estimated_effort,@created_at,@updated_at)',
    ).run(row);
    return row;
  },
  completeAssignment: (assignmentId: string) =>
    db
      .prepare("UPDATE study_assignments SET status='done', updated_at=? WHERE id=?")
      .run(now(), assignmentId),
  listAssignments: () => db.prepare('SELECT * FROM study_assignments ORDER BY due_date').all(),
});

export const jobRepo = (db: Database.Database) => ({
  createLead: (company: string, role: string) => {
    const ts = now();
    const row = {
      id: id('led'),
      company,
      role,
      source: 'manual',
      link: null,
      location: null,
      compensation: null,
      status: 'lead',
      notes: null,
      discovered_at: ts,
      created_at: ts,
      updated_at: ts,
    };
    db.prepare(
      'INSERT INTO job_leads (id,company,role,source,link,location,compensation,status,notes,discovered_at,created_at,updated_at) VALUES (@id,@company,@role,@source,@link,@location,@compensation,@status,@notes,@discovered_at,@created_at,@updated_at)',
    ).run(row);
    return row;
  },
  convertLeadToApplication: (leadId: string) => {
    const ts = now();
    const row = {
      id: id('app'),
      lead_id: leadId,
      resume_asset_id: null,
      cover_letter_ref: null,
      status: 'applied',
      applied_at: ts,
      follow_up_date: null,
      notes: null,
      created_at: ts,
      updated_at: ts,
    };
    db.prepare(
      'INSERT INTO job_applications (id,lead_id,resume_asset_id,cover_letter_ref,status,applied_at,follow_up_date,notes,created_at,updated_at) VALUES (@id,@lead_id,@resume_asset_id,@cover_letter_ref,@status,@applied_at,@follow_up_date,@notes,@created_at,@updated_at)',
    ).run(row);
    return row;
  },
  listLeads: () => db.prepare('SELECT * FROM job_leads ORDER BY discovered_at DESC').all(),
  listApplications: () =>
    db.prepare('SELECT * FROM job_applications ORDER BY created_at DESC').all(),
  createTask: (input: { title: string; description?: string }) => {
    const now = new Date().toISOString();
    const row: TaskRecord = {
      id: `tsk_${randomUUID().slice(0, 8)}`,
      title: input.title,
      description: input.description,
      status: 'todo',
      priority: 'medium',
      created_at: now,
      updated_at: now,
    };
    db.prepare(
      'INSERT INTO tasks (id,title,description,status,priority,created_at,updated_at) VALUES (@id,@title,@description,@status,@priority,@created_at,@updated_at)',
    ).run(row);
    return row;
  },
  listTasks: () => db.prepare('SELECT * FROM tasks ORDER BY created_at DESC').all() as TaskRecord[],
});

const parseMemoryRow = (row: Record<string, unknown>): MemoryRecord =>
  MemoryRecordSchema.parse({
    id: row.id,
    tier: row.tier,
    summary: row.summary,
    content: JSON.parse(String(row.content_json)),
    metadata: JSON.parse(String(row.metadata_json)),
    createdAt: row.created_at,
    updatedAt: row.updated_at,
    lastAccessedAt: row.last_accessed_at ?? undefined,
  });

export const memoryRepo = (db: Database.Database) => ({
  create: (input: MemoryWriteRequest): MemoryRecord => {
    const parsed = MemoryWriteRequestSchema.parse(input);
    const timestamp = now();
    const record = MemoryRecordSchema.parse({
      id: id('mem'),
      tier: parsed.tier,
      summary: parsed.summary,
      content: parsed.content,
      metadata: parsed.metadata,
      createdAt: timestamp,
      updatedAt: timestamp,
      lastAccessedAt: timestamp,
    });

    db.prepare(
      `INSERT INTO memories (id,tier,summary,content_json,metadata_json,created_at,updated_at,last_accessed_at)
       VALUES (@id,@tier,@summary,@content_json,@metadata_json,@created_at,@updated_at,@last_accessed_at)`,
    ).run({
      id: record.id,
      tier: record.tier,
      summary: record.summary,
      content_json: JSON.stringify(record.content),
      metadata_json: JSON.stringify(record.metadata),
      created_at: record.createdAt,
      updated_at: record.updatedAt,
      last_accessed_at: record.lastAccessedAt ?? null,
    });

    return record;
  },

  getById: (memoryId: string): MemoryRecord | null => {
    const row = db.prepare('SELECT * FROM memories WHERE id = ?').get(memoryId) as
      | Record<string, unknown>
      | undefined;

    if (!row) return null;

    const touchedAt = now();
    db.prepare('UPDATE memories SET last_accessed_at = ? WHERE id = ?').run(touchedAt, memoryId);

    return parseMemoryRow({
      ...row,
      last_accessed_at: touchedAt,
    });
  },

  list: (input?: MemoryReadRequest): MemoryRecord[] => {
    const parsed = MemoryReadRequestSchema.parse(input ?? {});
    const whereClauses: string[] = [];
    const params: Array<string | number> = [];

    if (parsed.tiers && parsed.tiers.length > 0) {
      const placeholders = parsed.tiers.map(() => '?').join(', ');
      whereClauses.push(`tier IN (${placeholders})`);
      params.push(...parsed.tiers);
    }

    if (parsed.minConfidence !== undefined) {
      whereClauses.push('CAST(json_extract(metadata_json, \'$.confidence\') AS REAL) >= ?');
      params.push(parsed.minConfidence);
    }

    if (parsed.text) {
      whereClauses.push(
        '(lower(summary) LIKE ? OR lower(content_json) LIKE ? OR lower(json_extract(metadata_json, \'$.tags\')) LIKE ?)',
      );
      const like = `%${parsed.text.trim().toLowerCase()}%`;
      params.push(like, like, like);
    }

    const whereSql = whereClauses.length > 0 ? `WHERE ${whereClauses.join(' AND ')}` : '';
    const rows = db
      .prepare(`SELECT * FROM memories ${whereSql} ORDER BY updated_at DESC LIMIT ?`)
      .all(...params, parsed.limit) as Array<Record<string, unknown>>;

    return rows.map(parseMemoryRow);
  },

  consolidate: (input: MemoryConsolidationRequest): MemoryRecord => {
    const parsed = MemoryConsolidationRequestSchema.parse(input);

    const missing = parsed.sourceMemoryIds.filter((sourceId) => {
      const row = db.prepare('SELECT id FROM memories WHERE id = ?').get(sourceId);
      return !row;
    });

    if (missing.length > 0) {
      throw new Error(`cannot consolidate missing memories: ${missing.join(', ')}`);
    }

    return memoryRepo(db).create({
      tier: parsed.targetTier,
      summary: parsed.summary,
      content: parsed.content,
      metadata: {
        ...parsed.metadata,
        provenance: {
          ...parsed.metadata.provenance,
          sourceMemoryIds: parsed.sourceMemoryIds,
          notes: parsed.reason,
        },
      },
    });
  },
});

const parseWorkspaceRow = (row: Record<string, unknown>): WorkspaceRecord =>
  WorkspaceRecordSchema.parse({
    id: row.id,
    slug: row.slug,
    name: row.name,
    repository: row.repository ?? undefined,
    metadata: JSON.parse(String(row.metadata_json)),
    state: JSON.parse(String(row.state_json)),
  });

const parseWorkspaceCheckpointRow = (row: Record<string, unknown>): WorkspaceCheckpoint =>
  WorkspaceCheckpointSchema.parse({
    id: row.id,
    workspaceId: row.workspace_id,
    summary: row.summary,
    snapshot: JSON.parse(String(row.snapshot_json)),
    createdAt: row.created_at,
    createdBy: row.created_by,
  });

const parseEnvironmentRow = (row: Record<string, unknown>): EnvironmentTwinRecord =>
  EnvironmentTwinRecordSchema.parse({
    id: row.id,
    slug: row.slug,
    name: row.name,
    metadata: JSON.parse(String(row.metadata_json)),
  });

const parseEnvironmentRecordRow = (row: Record<string, unknown>): EnvironmentRecord =>
  EnvironmentRecordSchema.parse({
    id: row.id,
    environmentId: row.environment_id,
    kind: row.kind,
    name: row.name,
    description: row.description ?? undefined,
    attributes: JSON.parse(String(row.attributes_json)),
    metadata: JSON.parse(String(row.metadata_json)),
  });

const parseEnvironmentRelationshipRow = (row: Record<string, unknown>): EnvironmentRelationship =>
  EnvironmentRelationshipSchema.parse({
    id: row.id,
    environmentId: row.environment_id,
    fromEntityId: row.from_entity_id,
    toEntityId: row.to_entity_id,
    relation: row.relation,
    direction: row.direction,
    notes: row.notes ?? undefined,
    createdAt: row.created_at,
  });

const parseEnvironmentAccessPathRow = (row: Record<string, unknown>): EnvironmentAccessPath =>
  EnvironmentAccessPathSchema.parse({
    id: row.id,
    environmentId: row.environment_id,
    entityId: row.entity_id,
    name: row.name,
    method: row.method,
    endpoint: row.endpoint,
    prerequisites: JSON.parse(String(row.prerequisites_json)),
    commandRefIds: JSON.parse(String(row.command_ref_ids_json)),
    notes: row.notes ?? undefined,
    createdAt: row.created_at,
    validatedAt: row.validated_at ?? undefined,
  });

const parseEnvironmentProcedureRow = (row: Record<string, unknown>): EnvironmentProcedure =>
  EnvironmentProcedureSchema.parse({
    id: row.id,
    environmentId: row.environment_id,
    name: row.name,
    intent: row.intent,
    targetEntityIds: JSON.parse(String(row.target_entity_ids_json)),
    steps: JSON.parse(String(row.steps_json)),
    lastValidatedAt: row.last_validated_at ?? undefined,
    owner: row.owner ?? undefined,
    tags: JSON.parse(String(row.tags_json)),
    createdAt: row.created_at,
    updatedAt: row.updated_at,
  });

export const workspaceRepo = (db: Database.Database) => ({
  create: (input: WorkspaceCreateRequest): WorkspaceRecord => {
    const parsed = WorkspaceCreateRequestSchema.parse(input);
    const existing = db.prepare('SELECT id FROM workspaces WHERE slug = ?').get(parsed.slug);
    if (existing) {
      throw new Error(`workspace slug already exists: ${parsed.slug}`);
    }

    const timestamp = now();
    const record = WorkspaceRecordSchema.parse({
      id: id('wrk'),
      slug: parsed.slug,
      name: parsed.name,
      repository: parsed.repository,
      metadata: {
        owner: parsed.metadata.owner,
        tags: parsed.metadata.tags ?? [],
        source: parsed.metadata.source ?? 'manual',
        createdAt: timestamp,
        updatedAt: timestamp,
        lastActivatedAt: parsed.metadata.lastActivatedAt,
      },
      state: {
        ...parsed.state,
        updatedAt: timestamp,
      },
    });

    db.prepare(
      `INSERT INTO workspaces (id,slug,name,repository,metadata_json,state_json,created_at,updated_at,last_activated_at)
       VALUES (@id,@slug,@name,@repository,@metadata_json,@state_json,@created_at,@updated_at,@last_activated_at)`,
    ).run({
      id: record.id,
      slug: record.slug,
      name: record.name,
      repository: record.repository ?? null,
      metadata_json: JSON.stringify(record.metadata),
      state_json: JSON.stringify(record.state),
      created_at: record.metadata.createdAt,
      updated_at: record.metadata.updatedAt,
      last_activated_at: record.metadata.lastActivatedAt ?? null,
    });

    return record;
  },

  list: (): WorkspaceSummary[] => {
    const rows = db
      .prepare('SELECT * FROM workspaces ORDER BY updated_at DESC')
      .all() as Array<Record<string, unknown>>;

    return rows.map((row) => {
      const workspace = parseWorkspaceRow(row);
      return WorkspaceSummarySchema.parse(deriveWorkspaceSummary(workspace));
    });
  },

  getBySlug: (slug: string): WorkspaceRecord | null => {
    const row = db.prepare('SELECT * FROM workspaces WHERE slug = ?').get(slug) as
      | Record<string, unknown>
      | undefined;
    if (!row) return null;
    return parseWorkspaceRow(row);
  },

  updateState: (workspaceId: string, patch: WorkspaceStatePatch): WorkspaceRecord | null => {
    const currentRow = db.prepare('SELECT * FROM workspaces WHERE id = ?').get(workspaceId) as
      | Record<string, unknown>
      | undefined;
    if (!currentRow) return null;

    const parsedPatch = WorkspaceStatePatchSchema.parse(patch);
    const current = parseWorkspaceRow(currentRow);
    const timestamp = now();
    const updated = WorkspaceRecordSchema.parse({
      ...current,
      metadata: {
        ...current.metadata,
        updatedAt: timestamp,
      },
      state: {
        ...current.state,
        ...parsedPatch,
        updatedAt: timestamp,
      },
    });

    db.prepare(
      `UPDATE workspaces
       SET state_json = @state_json,
           metadata_json = @metadata_json,
           updated_at = @updated_at,
           last_activated_at = @last_activated_at
       WHERE id = @id`,
    ).run({
      id: updated.id,
      state_json: JSON.stringify(updated.state),
      metadata_json: JSON.stringify(updated.metadata),
      updated_at: updated.metadata.updatedAt,
      last_activated_at: updated.metadata.lastActivatedAt ?? null,
    });

    return updated;
  },

  setActivated: (workspaceId: string): WorkspaceRecord | null => {
    const currentRow = db.prepare('SELECT * FROM workspaces WHERE id = ?').get(workspaceId) as
      | Record<string, unknown>
      | undefined;
    if (!currentRow) return null;

    const current = parseWorkspaceRow(currentRow);
    const timestamp = now();
    const updated = WorkspaceRecordSchema.parse({
      ...current,
      metadata: {
        ...current.metadata,
        updatedAt: timestamp,
        lastActivatedAt: timestamp,
      },
    });

    db.prepare(
      `UPDATE workspaces
       SET metadata_json = @metadata_json,
           updated_at = @updated_at,
           last_activated_at = @last_activated_at
       WHERE id = @id`,
    ).run({
      id: updated.id,
      metadata_json: JSON.stringify(updated.metadata),
      updated_at: updated.metadata.updatedAt,
      last_activated_at: updated.metadata.lastActivatedAt ?? null,
    });

    return updated;
  },

  createCheckpoint: (
    workspaceId: string,
    input: WorkspaceCheckpointCreateRequest,
  ): WorkspaceCheckpoint | null => {
    const currentRow = db.prepare('SELECT * FROM workspaces WHERE id = ?').get(workspaceId) as
      | Record<string, unknown>
      | undefined;
    if (!currentRow) return null;

    const current = parseWorkspaceRow(currentRow);
    const parsed = WorkspaceCheckpointCreateRequestSchema.parse(input);
    const snapshot = {
      ...(parsed.snapshot ?? current.state),
      updatedAt: now(),
    };

    const checkpoint = WorkspaceCheckpointSchema.parse({
      id: id('wcp'),
      workspaceId,
      summary: parsed.summary,
      snapshot,
      createdAt: now(),
      createdBy: parsed.createdBy,
    });

    db.prepare(
      `INSERT INTO workspace_checkpoints (id,workspace_id,summary,snapshot_json,created_at,created_by)
       VALUES (@id,@workspace_id,@summary,@snapshot_json,@created_at,@created_by)`,
    ).run({
      id: checkpoint.id,
      workspace_id: checkpoint.workspaceId,
      summary: checkpoint.summary,
      snapshot_json: JSON.stringify(checkpoint.snapshot),
      created_at: checkpoint.createdAt,
      created_by: checkpoint.createdBy,
    });

    return checkpoint;
  },

  listCheckpoints: (workspaceId: string, limit = 10): WorkspaceCheckpoint[] => {
    const parsedLimit = normalizeWorkspaceCheckpointLimit(limit, 10);
    const rows = db
      .prepare(
        'SELECT * FROM workspace_checkpoints WHERE workspace_id = ? ORDER BY created_at DESC LIMIT ?',
      )
      .all(workspaceId, parsedLimit) as Array<Record<string, unknown>>;

    return rows.map(parseWorkspaceCheckpointRow);
  },
});

export const environmentRepo = (db: Database.Database) => ({
  create: (input: EnvironmentTwinCreateRequest): EnvironmentTwinRecord => {
    const parsed = EnvironmentTwinCreateRequestSchema.parse(input);
    const existing = db.prepare('SELECT id FROM environments WHERE slug = ?').get(parsed.slug);
    if (existing) {
      throw new Error(`environment slug already exists: ${parsed.slug}`);
    }

    const timestamp = now();
    const record = EnvironmentTwinRecordSchema.parse({
      id: id('env'),
      slug: parsed.slug,
      name: parsed.name,
      metadata: {
        owner: parsed.metadata.owner,
        tags: parsed.metadata.tags ?? [],
        source: parsed.metadata.source ?? 'manual',
        createdAt: timestamp,
        updatedAt: timestamp,
        lastValidatedAt: parsed.metadata.lastValidatedAt,
      },
    });

    db.prepare(
      `INSERT INTO environments (id,slug,name,metadata_json,created_at,updated_at,last_validated_at)
       VALUES (@id,@slug,@name,@metadata_json,@created_at,@updated_at,@last_validated_at)`,
    ).run({
      id: record.id,
      slug: record.slug,
      name: record.name,
      metadata_json: JSON.stringify(record.metadata),
      created_at: record.metadata.createdAt,
      updated_at: record.metadata.updatedAt,
      last_validated_at: record.metadata.lastValidatedAt ?? null,
    });

    return record;
  },

  list: (): EnvironmentSummary[] => {
    const rows = db
      .prepare('SELECT * FROM environments ORDER BY updated_at DESC')
      .all() as Array<Record<string, unknown>>;

    return rows.map((row) => {
      const env = parseEnvironmentRow(row);
      const recordCount =
        (db
          .prepare('SELECT COUNT(*) as count FROM environment_records WHERE environment_id = ?')
          .get(env.id) as { count: number })?.count ?? 0;
      const relationshipCount =
        (db
          .prepare('SELECT COUNT(*) as count FROM environment_relationships WHERE environment_id = ?')
          .get(env.id) as { count: number })?.count ?? 0;
      const procedureCount =
        (db
          .prepare('SELECT COUNT(*) as count FROM environment_procedures WHERE environment_id = ?')
          .get(env.id) as { count: number })?.count ?? 0;

      return EnvironmentSummarySchema.parse({
        id: env.id,
        slug: env.slug,
        name: env.name,
        owner: env.metadata.owner,
        recordCount,
        relationshipCount,
        procedureCount,
        updatedAt: env.metadata.updatedAt,
        lastValidatedAt: env.metadata.lastValidatedAt,
      });
    });
  },

  getBySlug: (slug: string): EnvironmentTwinRecord | null => {
    const row = db.prepare('SELECT * FROM environments WHERE slug = ?').get(slug) as
      | Record<string, unknown>
      | undefined;
    if (!row) return null;
    return parseEnvironmentRow(row);
  },

  upsertRecord: (environmentId: string, record: EnvironmentRecord): EnvironmentRecord | null => {
    const envRow = db.prepare('SELECT * FROM environments WHERE id = ?').get(environmentId) as
      | Record<string, unknown>
      | undefined;
    if (!envRow) return null;

    const parsed = EnvironmentRecordSchema.parse(record);
    db.prepare(
      `INSERT INTO environment_records
       (id,environment_id,kind,name,description,attributes_json,metadata_json,created_at,updated_at,validated_at)
       VALUES (@id,@environment_id,@kind,@name,@description,@attributes_json,@metadata_json,@created_at,@updated_at,@validated_at)
       ON CONFLICT(id) DO UPDATE SET
         kind=excluded.kind,
         name=excluded.name,
         description=excluded.description,
         attributes_json=excluded.attributes_json,
         metadata_json=excluded.metadata_json,
         updated_at=excluded.updated_at,
         validated_at=excluded.validated_at`,
    ).run({
      id: parsed.id,
      environment_id: parsed.environmentId,
      kind: parsed.kind,
      name: parsed.name,
      description: parsed.description ?? null,
      attributes_json: JSON.stringify(parsed.attributes),
      metadata_json: JSON.stringify(parsed.metadata),
      created_at: parsed.metadata.createdAt,
      updated_at: parsed.metadata.updatedAt,
      validated_at: parsed.metadata.validatedAt ?? null,
    });

    const env = parseEnvironmentRow(envRow);
    const updatedMeta = {
      ...env.metadata,
      updatedAt: now(),
    };
    db.prepare(
      `UPDATE environments
       SET metadata_json = @metadata_json,
           updated_at = @updated_at,
           last_validated_at = @last_validated_at
       WHERE id = @id`,
    ).run({
      id: env.id,
      metadata_json: JSON.stringify(updatedMeta),
      updated_at: updatedMeta.updatedAt,
      last_validated_at: updatedMeta.lastValidatedAt ?? null,
    });

    const row = db.prepare('SELECT * FROM environment_records WHERE id = ?').get(parsed.id) as
      | Record<string, unknown>
      | undefined;
    if (!row) return null;
    return parseEnvironmentRecordRow(row);
  },

  listRecords: (environmentId: string, limit = 50): EnvironmentRecord[] => {
    const parsedLimit = normalizeEnvironmentItemLimit(limit, 50, 200);
    const rows = db
      .prepare(
        'SELECT * FROM environment_records WHERE environment_id = ? ORDER BY updated_at DESC LIMIT ?',
      )
      .all(environmentId, parsedLimit) as Array<Record<string, unknown>>;

    return rows.map(parseEnvironmentRecordRow);
  },

  addRelationship: (
    environmentId: string,
    input: EnvironmentRelationship,
  ): EnvironmentRelationship | null => {
    const env = db.prepare('SELECT id FROM environments WHERE id = ?').get(environmentId);
    if (!env) return null;

    const parsed = EnvironmentRelationshipSchema.parse(input);
    db.prepare(
      `INSERT INTO environment_relationships
       (id,environment_id,from_entity_id,to_entity_id,relation,direction,notes,created_at)
       VALUES (@id,@environment_id,@from_entity_id,@to_entity_id,@relation,@direction,@notes,@created_at)`,
    ).run({
      id: parsed.id,
      environment_id: parsed.environmentId,
      from_entity_id: parsed.fromEntityId,
      to_entity_id: parsed.toEntityId,
      relation: parsed.relation,
      direction: parsed.direction,
      notes: parsed.notes ?? null,
      created_at: parsed.createdAt,
    });

    return parsed;
  },

  listRelationships: (environmentId: string, limit = 50): EnvironmentRelationship[] => {
    const parsedLimit = normalizeEnvironmentItemLimit(limit, 50, 200);
    const rows = db
      .prepare(
        'SELECT * FROM environment_relationships WHERE environment_id = ? ORDER BY created_at DESC LIMIT ?',
      )
      .all(environmentId, parsedLimit) as Array<Record<string, unknown>>;
    return rows.map(parseEnvironmentRelationshipRow);
  },

  addAccessPath: (environmentId: string, input: EnvironmentAccessPath): EnvironmentAccessPath | null => {
    const env = db.prepare('SELECT id FROM environments WHERE id = ?').get(environmentId);
    if (!env) return null;

    const parsed = EnvironmentAccessPathSchema.parse(input);
    db.prepare(
      `INSERT INTO environment_access_paths
       (id,environment_id,entity_id,name,method,endpoint,prerequisites_json,command_ref_ids_json,notes,created_at,validated_at)
       VALUES (@id,@environment_id,@entity_id,@name,@method,@endpoint,@prerequisites_json,@command_ref_ids_json,@notes,@created_at,@validated_at)`,
    ).run({
      id: parsed.id,
      environment_id: parsed.environmentId,
      entity_id: parsed.entityId,
      name: parsed.name,
      method: parsed.method,
      endpoint: parsed.endpoint,
      prerequisites_json: JSON.stringify(parsed.prerequisites),
      command_ref_ids_json: JSON.stringify(parsed.commandRefIds),
      notes: parsed.notes ?? null,
      created_at: parsed.createdAt,
      validated_at: parsed.validatedAt ?? null,
    });

    return parsed;
  },

  listAccessPaths: (environmentId: string, limit = 50): EnvironmentAccessPath[] => {
    const parsedLimit = normalizeEnvironmentItemLimit(limit, 50, 200);
    const rows = db
      .prepare(
        'SELECT * FROM environment_access_paths WHERE environment_id = ? ORDER BY created_at DESC LIMIT ?',
      )
      .all(environmentId, parsedLimit) as Array<Record<string, unknown>>;
    return rows.map(parseEnvironmentAccessPathRow);
  },

  addProcedure: (environmentId: string, input: EnvironmentProcedure): EnvironmentProcedure | null => {
    const env = db.prepare('SELECT id FROM environments WHERE id = ?').get(environmentId);
    if (!env) return null;

    const parsed = EnvironmentProcedureSchema.parse(input);
    db.prepare(
      `INSERT INTO environment_procedures
       (id,environment_id,name,intent,target_entity_ids_json,steps_json,last_validated_at,owner,tags_json,created_at,updated_at)
       VALUES (@id,@environment_id,@name,@intent,@target_entity_ids_json,@steps_json,@last_validated_at,@owner,@tags_json,@created_at,@updated_at)`,
    ).run({
      id: parsed.id,
      environment_id: parsed.environmentId,
      name: parsed.name,
      intent: parsed.intent,
      target_entity_ids_json: JSON.stringify(parsed.targetEntityIds),
      steps_json: JSON.stringify(parsed.steps),
      last_validated_at: parsed.lastValidatedAt ?? null,
      owner: parsed.owner ?? null,
      tags_json: JSON.stringify(parsed.tags),
      created_at: parsed.createdAt,
      updated_at: parsed.updatedAt,
    });

    return parsed;
  },

  listProcedures: (environmentId: string, limit = 50): EnvironmentProcedure[] => {
    const parsedLimit = normalizeEnvironmentItemLimit(limit, 50, 200);
    const rows = db
      .prepare(
        'SELECT * FROM environment_procedures WHERE environment_id = ? ORDER BY updated_at DESC LIMIT ?',
      )
      .all(environmentId, parsedLimit) as Array<Record<string, unknown>>;
    return rows.map(parseEnvironmentProcedureRow);
  },
});

export const eventRepo = (db: Database.Database) => ({
  record: (event: unknown) => {
    const parsed = DomainEventSchema.parse(event);
    db.prepare(
      `INSERT INTO system_events (event_id,event_type,actor,timestamp,domain,entity_type,entity_id,payload_json,sensitivity,trace_id)
       VALUES (@eventId,@eventType,@actor,@timestamp,@domain,@entityType,@entityId,@payload,@sensitivity,@traceId)`,
    ).run({
      ...parsed,
      entityType: parsed.entityRef.type,
      entityId: parsed.entityRef.id,
      payload: JSON.stringify(parsed.payload),
    });
  },
  list: (limit = 100) =>
    db.prepare('SELECT * FROM system_events ORDER BY timestamp DESC LIMIT ?').all(limit),
});
