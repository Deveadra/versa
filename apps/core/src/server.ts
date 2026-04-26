import express, { Request, Response } from 'express';
import cors from 'cors';
import { randomUUID } from 'node:crypto';
import { loadConfig } from '@versa/config';
import {
  connectDb,
  eventRepo,
  goalRepo,
  jobRepo,
  scheduleRepo,
  studyRepo,
  taskRepo,
} from '@versa/database';
import { log } from '@versa/logging';
import { generateDailyPlan } from './planner';

const app = express();
const cfg = loadConfig();
const db = connectDb(cfg.DATABASE_URL);
const tasks = taskRepo(db);
const goals = goalRepo(db);
const schedules = scheduleRepo(db);
const study = studyRepo(db);
const jobs = jobRepo(db);
const events = eventRepo(db);

const asString = (value: unknown) =>
  Array.isArray(value) ? String(value[0]) : String(value ?? '');

const emit = (
  eventType: string,
  entityType: string,
  entityId: string,
  payload: Record<string, unknown>,
  domain: 'core' | 'study' | 'jobs' = 'core',
) => {
  events.record({
    eventId: `evt_${randomUUID().slice(0, 8)}`,
    eventType,
    actor: 'core-api',
    timestamp: new Date().toISOString(),
    domain,
    entityRef: { type: entityType, id: entityId },
    payload,
    sensitivity: 'internal',
    traceId: randomUUID(),
  });
};

const getDailyPlan = () => {
  const overdueTasks = tasks.list('overdue') as Array<{ title: string }>;
  const todayTasks = tasks.list('today') as Array<{ title: string }>;
  const todayBlocks = schedules.listDay(new Date().toISOString().slice(0, 10)) as Array<{
    title: string;
    status: string;
  }>;
  const activeGoals = (goals.list() as Array<{ title: string; status: string }>).filter(
    (goal) => goal.status === 'active',
  );
  const followUpsSoon = (jobs.listApplications() as Array<{ follow_up_date?: string }>).filter(
    (appRow) => !!appRow.follow_up_date,
  ).length;
  const studyPendingCount = (study.listAssignments() as Array<{ status: string }>).filter(
    (assignment) => assignment.status !== 'done',
  ).length;

  return generateDailyPlan({
    overdueTasks,
    todayTasks,
    activeGoals,
    todayBlocks,
    studyPendingCount,
    followUpsSoon,
  });
};

app.use(cors());
app.use(express.json());

app.get('/health', (_req: Request, res: Response) => res.json({ ok: true }));

app.get('/tasks', (req: Request, res: Response) => {
  const scope = (req.query.scope as 'all' | 'today' | 'overdue' | undefined) ?? 'all';
  res.json({ data: tasks.list(scope) });
});

app.post('/tasks', (req: Request, res: Response) => {
  const task = tasks.create(req.body as Record<string, unknown> as any);
  emit('task.created', 'task', String(task.id), { title: task.title });
  res.status(201).json({ data: task });
});

app.patch('/tasks/:taskId', (req: Request, res: Response) => {
  const updated = tasks.update(asString(req.params.taskId), req.body as Record<string, unknown>);
  if (!updated) return res.status(404).json({ error: 'task not found' });
  const eventType =
    (updated as Record<string, unknown>).status === 'done' ? 'task.completed' : 'task.updated';
  emit(eventType, 'task', asString(req.params.taskId), req.body as Record<string, unknown>);
  return res.json({ data: updated });
});

app.delete('/tasks/:taskId', (req: Request, res: Response) => {
  tasks.remove(asString(req.params.taskId));
  emit('task.archived', 'task', asString(req.params.taskId), {});
  res.status(204).send();
});

app.get('/goals', (_req: Request, res: Response) => res.json({ data: goals.list() }));
app.post('/goals', (req: Request, res: Response) => {
  const goal = goals.create(req.body);
  emit('goal.created', 'goal', String(goal.id), { title: goal.title });
  res.status(201).json({ data: goal });
});
app.patch('/goals/:goalId', (req: Request, res: Response) => {
  const updated = goals.update(asString(req.params.goalId), req.body as Record<string, unknown>);
  if (!updated) return res.status(404).json({ error: 'goal not found' });
  emit(
    (updated as Record<string, unknown>).status === 'completed' ? 'goal.completed' : 'goal.updated',
    'goal',
    asString(req.params.goalId),
    req.body as Record<string, unknown>,
  );
  return res.json({ data: updated });
});

app.get('/schedule', (req: Request, res: Response) => {
  const view = (req.query.view as 'day' | 'week' | undefined) ?? 'day';
  const date = (req.query.date as string | undefined) ?? new Date().toISOString().slice(0, 10);
  res.json({ data: view === 'week' ? schedules.listWeek() : schedules.listDay(date) });
});
app.post('/schedule', (req: Request, res: Response) => {
  try {
    const block = schedules.create(req.body);
    emit('schedule.block.created', 'schedule_block', String(block.id), { title: block.title });
    res.status(201).json({ data: block });
  } catch (error) {
    res.status(400).json({ error: (error as Error).message });
  }
});
app.patch('/schedule/:blockId/status', (req: Request, res: Response) => {
  schedules.setStatus(asString(req.params.blockId), asString(req.body.status ?? 'scheduled'));
  const status = asString(req.body.status);
  emit(
    status === 'completed'
      ? 'schedule.block.completed'
      : status === 'missed'
        ? 'schedule.block.missed'
        : 'schedule.block.updated',
    'schedule_block',
    asString(req.params.blockId),
    { status },
  );
  res.json({ ok: true });
});

app.get('/study/assignments', (_req: Request, res: Response) =>
  res.json({ data: study.listAssignments() }),
);
app.post('/study/courses', (req: Request, res: Response) => {
  const course = study.createCourse(String(req.body.title));
  emit('study.course.created', 'study_course', String(course.id), { title: course.title }, 'study');
  res.status(201).json({ data: course });
});
app.post('/study/assignments', (req: Request, res: Response) => {
  const assignment = study.createAssignment(
    String(req.body.courseId),
    String(req.body.title),
    req.body.dueDate,
  );
  emit(
    'study.assignment.created',
    'study_assignment',
    String(assignment.id),
    { title: assignment.title },
    'study',
  );
  res.status(201).json({ data: assignment });
});
app.patch('/study/assignments/:assignmentId/complete', (req: Request, res: Response) => {
  study.completeAssignment(asString(req.params.assignmentId));
  emit(
    'study.assignment.completed',
    'study_assignment',
    asString(req.params.assignmentId),
    {},
    'study',
  );
  res.json({ ok: true });
});

app.get('/jobs', (_req: Request, res: Response) =>
  res.json({ data: { leads: jobs.listLeads(), applications: jobs.listApplications() } }),
);
app.post('/jobs/leads', (req: Request, res: Response) => {
  const lead = jobs.createLead(String(req.body.company), String(req.body.role));
  emit('job.lead.created', 'job_lead', String(lead.id), { company: lead.company }, 'jobs');
  res.status(201).json({ data: lead });
});
app.post('/jobs/leads/:leadId/convert', (req: Request, res: Response) => {
  const application = jobs.convertLeadToApplication(asString(req.params.leadId));
  emit(
    'job.application.created',
    'job_application',
    String(application.id),
    { leadId: asString(req.params.leadId) },
    'jobs',
  );
  res.status(201).json({ data: application });
});

app.get('/planner/today', (_req: Request, res: Response) => res.json({ data: getDailyPlan() }));
app.get('/events', (req: Request, res: Response) => {
  const limit = Number(req.query.limit ?? 100);
  res.json({ data: events.list(limit) });
});

app.post('/quick-capture', (req: Request, res: Response) => {
  const kind = String(req.body.kind);
  if (kind === 'task')
    return res.status(201).json({ data: tasks.create({ title: String(req.body.title) }) });
  if (kind === 'goal')
    return res.status(201).json({ data: goals.create({ title: String(req.body.title) }) });
  if (kind === 'schedule') return res.status(201).json({ data: schedules.create(req.body) });
  if (kind === 'study-assignment')
    return res
      .status(201)
      .json({ data: study.createAssignment(String(req.body.courseId), String(req.body.title)) });
  if (kind === 'job-lead')
    return res
      .status(201)
      .json({ data: jobs.createLead(String(req.body.company), String(req.body.role)) });
  return res.status(400).json({ error: 'unsupported quick capture kind' });
});

app.post('/scheduler/run', (_req: Request, res: Response) =>
  res.json({ ok: true, message: 'scheduler stub' }),
);
app.get('/integrations', (_req: Request, res: Response) =>
  res.json({ providers: ['google', 'notion', 'github'] }),
);
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
