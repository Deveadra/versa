import Database from 'better-sqlite3';
import { randomUUID } from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { DomainEventSchema } from '@versa/shared';

const defaultDbPath = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../data/versa.db');

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
});
