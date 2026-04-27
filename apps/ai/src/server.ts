import { randomUUID } from 'node:crypto';
import express, { Request, Response } from 'express';
import { loadConfig } from '@versa/config';
import { classifySkillExecution, createApprovalPolicyEngine } from '@versa/approvals';
import {
  createLogger,
  getTelemetryContext,
  createNdjsonFileSink,
  createRequestTelemetryMiddleware,
  createTelemetrySink,
} from '@versa/logging';
import { createFoundationalSkillRegistry, type SkillDefinition } from '@versa/skills';
import { TrustLevelEnum } from '@versa/shared';

const app = express();
const cfg = loadConfig();

const telemetryFileSink = createNdjsonFileSink('artifacts/telemetry.ndjson');
const skillRegistry = createFoundationalSkillRegistry();
const approvalPolicy = createApprovalPolicyEngine();
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
    data: skillRegistry.list().map((skill: SkillDefinition) => ({
      id: skill.id,
      name: skill.name,
      metadata: skill.metadata,
    })),
  }),
);

app.post('/skills/execute', (req: Request, res: Response) => {
  const requestId = randomUUID();
  const trace = getTelemetryContext(req);

  if (cfg.FEATURE_APPROVALS_ENABLED) {
    const skillId = typeof req.body.skillId === 'string' ? req.body.skillId : undefined;
    const skillName = typeof req.body.skillName === 'string' ? req.body.skillName : undefined;
    const risky =
      req.body?.input &&
      typeof req.body.input === 'object' &&
      ((req.body.input as Record<string, unknown>).risky === true ||
        (req.body.input as Record<string, unknown>).destructive === true);
    const trustLevelInput =
      req.body?.context && typeof req.body.context === 'object'
        ? (req.body.context as Record<string, unknown>).trustLevel
        : undefined;
    const parsedTrustLevel = TrustLevelEnum.safeParse(trustLevelInput);
    const trustLevel = parsedTrustLevel.success ? parsedTrustLevel.data : 'safe-act';

    const approval = approvalPolicy.decide({
      requestId,
      requestedAt: new Date().toISOString(),
      actor: 'ai-service',
      trustLevel,
      action: 'skills.execute',
      classification: classifySkillExecution({
        action: 'skills.execute',
        skillId,
        skillName,
        risky: risky === true,
      }),
      audit: {
        traceId: trace.traceId ?? requestId,
        correlationId: trace.correlationId,
        requestId: trace.requestId,
        source: 'apps.ai',
        timestamp: new Date().toISOString(),
      },
      context: {
        skillId,
        skillName,
      },
    });

    logger.info(
      'approval.policy.evaluated',
      'approval policy evaluated for skills.execute',
      {
        requestId,
        outcome: approval.result.outcome,
        policyRuleId: approval.result.policyRuleId,
        requiresApproval: approval.result.requiresApproval,
      },
      trace,
    );

    if (approval.result.outcome === 'deny') {
      return res.status(403).json({
        error: 'skill execution denied by approval policy',
        approval: approval.result,
      });
    }

    if (approval.result.outcome === 'require_approval') {
      return res.status(409).json({
        error: 'skill execution requires operator approval',
        approval: approval.result,
      });
    }
  }

  const result = skillRegistry.execute({
    skillId: typeof req.body.skillId === 'string' ? req.body.skillId : undefined,
    skillName: typeof req.body.skillName === 'string' ? req.body.skillName : undefined,
    input:
      req.body.input && typeof req.body.input === 'object'
        ? (req.body.input as Record<string, unknown>)
        : {},
    context: {},
  });

  if (result.status === 'succeeded') return res.status(200).json({ data: result });
  if (result.status === 'invalid_request') return res.status(422).json({ data: result });
  if (result.error?.code === 'SKILL_NOT_FOUND') return res.status(404).json({ data: result });
  if (result.status === 'blocked') return res.status(409).json({ data: result });
  return res.status(500).json({ data: result });
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
