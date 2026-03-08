import express, { Request, Response } from 'express';
import cors from 'cors';
import { randomUUID } from 'node:crypto';
import { loadConfig } from '@versa/config';
import { connectDb, eventRepo, taskRepo } from '@versa/database';
import { log } from '@versa/logging';

const app = express();
const cfg = loadConfig();
const db = connectDb(cfg.DATABASE_URL);
const tasks = taskRepo(db);
const events = eventRepo(db);

app.use(cors());
app.use(express.json());

app.get('/health', (_req: Request, res: Response) => res.json({ ok: true }));

app.get('/tasks', (_req: Request, res: Response) => {
  res.json({ data: tasks.listTasks() });
});

app.post('/tasks', (req: Request, res: Response) => {
  const task = tasks.createTask({ title: String(req.body.title), description: req.body.description });
  events.record({
    eventId: `evt_${randomUUID().slice(0, 8)}`,
    eventType: 'task.created',
    actor: 'core-api',
    timestamp: new Date().toISOString(),
    domain: 'core',
    entityRef: { type: 'task', id: task.id },
    payload: { title: task.title },
    sensitivity: 'internal',
    traceId: randomUUID(),
  });
  log('info', 'task.created', { taskId: task.id });
  res.status(201).json({ data: task });
});

app.post('/scheduler/run', (_req: Request, res: Response) => res.json({ ok: true, message: 'scheduler stub' }));
app.get('/integrations', (_req: Request, res: Response) => res.json({ providers: ['google', 'notion', 'github'] }));
app.get('/ai/health', async (_req: Request, res: Response) => {
  try {
    const response = await fetch(`http://localhost:${cfg.AI_PORT}/health`);
    res.json({ ok: response.ok });
  } catch {
    res.status(200).json({ ok: false, fallback: 'core continues without ai' });
  }
});

app.listen(cfg.CORE_PORT, () => {
  log('info', 'core.started', { port: cfg.CORE_PORT });
});
