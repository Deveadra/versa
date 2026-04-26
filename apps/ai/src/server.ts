import express, { Request, Response } from 'express';
import { loadConfig } from '@versa/config';
import {
  createLogger,
  createNdjsonFileSink,
  createRequestTelemetryMiddleware,
  createTelemetrySink,
} from '@versa/logging';
import { createFoundationalSkillRegistry } from '@versa/skills';

const app = express();
const cfg = loadConfig();

const telemetryFileSink = createNdjsonFileSink('artifacts/telemetry.ndjson');
const skillRegistry = createFoundationalSkillRegistry();
const logger = createLogger({
  actor: {
    service: 'ai',
    source: 'http',
  },
  sink: createTelemetrySink({
    consoleEnabled: cfg.TELEMETRY_CONSOLE_ENABLED,
    sinks: cfg.TELEMETRY_ENABLED ? [telemetryFileSink] : [],
  }),
});

const capabilities = {
  summarize_day: () => ({ summary: 'placeholder summary' }),
  generate_study_plan: () => ({ plan: [] }),
  rank_priorities: () => ({ priorities: [] }),
  record_memory: () => ({ ok: true }),
  search_memory: () => ({ results: [] }),
};

app.use(express.json());
app.use(createRequestTelemetryMiddleware(logger));

app.get('/health', (_req: Request, res: Response) => res.json({ ok: true }));
app.get('/capabilities', (_req: Request, res: Response) => res.json({ capabilities: Object.keys(capabilities) }));
app.get('/skills', (_req: Request, res: Response) =>
  res.json({
    data: skillRegistry.list().map((skill: { id: string; name: string; metadata: unknown }) => ({
      id: skill.id,
      name: skill.name,
      metadata: skill.metadata,
    })),
  }),
);

app.post('/skills/execute', (req: Request, res: Response) => {
  const result = skillRegistry.execute({
    skillId: typeof req.body.skillId === 'string' ? req.body.skillId : undefined,
    skillName: typeof req.body.skillName === 'string' ? req.body.skillName : undefined,
    input:
      req.body.input && typeof req.body.input === 'object'
        ? (req.body.input as Record<string, unknown>)
        : {},
    context: {},
  });

  if (result.status === 'succeeded') {
    return res.status(200).json({ data: result });
  }

  return res.status(400).json({ data: result });
});

const server = app.listen(cfg.AI_PORT, () => {
  logger.info('app.started', 'ai server started', { port: cfg.AI_PORT });
});

const handleShutdown = (signal: string) => {
  logger.info('app.shutdown.requested', 'ai shutdown requested', { signal });
  server.close(() => {
    logger.info('app.stopped', 'ai server stopped', { signal });
  });
};

process.once('SIGINT', () => handleShutdown('SIGINT'));
process.once('SIGTERM', () => handleShutdown('SIGTERM'));
