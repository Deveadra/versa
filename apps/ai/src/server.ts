import express, { Request, Response } from 'express';
import { loadConfig } from '@versa/config';
import {
  createLogger,
  createNdjsonFileSink,
  createRequestTelemetryMiddleware,
} from '@versa/logging';

const app = express();
const cfg = loadConfig();

const telemetryFileSink = createNdjsonFileSink('artifacts/telemetry.ndjson');
const logger = createLogger({
  actor: {
    service: 'ai',
    source: 'http',
  },
  sink: (event) => {
    if (cfg.TELEMETRY_CONSOLE_ENABLED) {
      console.log(JSON.stringify(event));
    }

    if (cfg.TELEMETRY_ENABLED) {
      telemetryFileSink(event);
    }
  },
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
