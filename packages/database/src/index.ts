import Database from 'better-sqlite3';
import { randomUUID } from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  DomainEventSchema,
  MemoryConsolidationRequestSchema,
  MemoryReadRequestSchema,
  MemoryRecordSchema,
  MemoryWriteRequestSchema,
  type MemoryConsolidationRequest,
  type MemoryReadRequest,
  type MemoryRecord,
  type MemoryWriteRequest,
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
    const rows = db.prepare('SELECT * FROM memories ORDER BY updated_at DESC LIMIT ?').all(parsed.limit) as Array<
      Record<string, unknown>
    >;

    let items = rows.map(parseMemoryRow);

    if (parsed.tiers && parsed.tiers.length > 0) {
      const tiers = new Set(parsed.tiers);
      items = items.filter((item) => tiers.has(item.tier));
    }

    if (parsed.minConfidence !== undefined) {
      const minConfidence = parsed.minConfidence;
      items = items.filter((item) => item.metadata.confidence >= minConfidence);
    }

    if (parsed.text) {
      const q = parsed.text.trim().toLowerCase();
      items = items.filter((item) => {
        const summary = item.summary.toLowerCase();
        const content = JSON.stringify(item.content).toLowerCase();
        const tags = item.metadata.tags.join(' ').toLowerCase();
        return summary.includes(q) || content.includes(q) || tags.includes(q);
      });
    }

    return items;
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
