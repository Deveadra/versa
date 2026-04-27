import { randomUUID } from 'node:crypto';
import express, { Request, Response } from 'express';
import { createBridgeAdapter } from '@versa/bridge';
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
import {
  AiServiceRequestSchema,
  AiServiceResponseSchema,
  LegacyBridgeRequestSchema,
  TrustLevelEnum,
} from '@versa/shared';

const app = express();
const cfg = loadConfig();

const telemetryFileSink = createNdjsonFileSink('artifacts/telemetry.ndjson');
const skillRegistry = createFoundationalSkillRegistry();
const approvalPolicy = createApprovalPolicyEngine();
const bridge = createBridgeAdapter({
  BRIDGE_ENABLED: cfg.BRIDGE_ENABLED,
  BRIDGE_MODE: cfg.BRIDGE_MODE,
  BRIDGE_LEGACY_RUNTIME_URL: cfg.BRIDGE_LEGACY_RUNTIME_URL,
  BRIDGE_HEALTH_PATH: cfg.BRIDGE_HEALTH_PATH,
  BRIDGE_CAPABILITIES_PATH: cfg.BRIDGE_CAPABILITIES_PATH,
  BRIDGE_INVOKE_PATH: cfg.BRIDGE_INVOKE_PATH,
});
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

const shouldUseLegacyBridge = (capabilityId: string, requestedTarget?: string) =>
  requestedTarget === 'legacy_python_runtime' || capabilityId.startsWith('legacy.');

app.use(express.json());
app.use(createRequestTelemetryMiddleware(logger));

app.get('/health', (_req: Request, res: Response) => res.json({ ok: true }));
app.get('/capabilities', (_req: Request, res: Response) => res.json({ capabilities: Object.keys(capabilities) }));

app.get('/bridge/health', (_req: Request, res: Response) => {
  const data = bridge.health();
  res.json({ data });
});

app.get('/bridge/capabilities', (_req: Request, res: Response) => {
  const data = bridge.capabilities();
  res.json({ data });
});

app.post('/bridge/invoke', (req: Request, res: Response) => {
  const requestId = randomUUID();
  const trace = getTelemetryContext(req);

  const parsedRequest = LegacyBridgeRequestSchema.parse({
    requestId,
    operation: 'invoke',
    capabilityId: typeof req.body.capabilityId === 'string' ? req.body.capabilityId : undefined,
    payload:
      req.body.payload && typeof req.body.payload === 'object'
        ? (req.body.payload as Record<string, unknown>)
        : {},
    context: {
      traceId: trace.traceId,
      correlationId: trace.correlationId,
      requestId: trace.requestId,
      actor: 'ai-service',
      source: 'apps.ai',
    },
  });

  if (cfg.FEATURE_APPROVALS_ENABLED) {
    const risky =
      parsedRequest.payload.risky === true || parsedRequest.payload.destructive === true;
    const approval = approvalPolicy.decide({
      requestId,
      requestedAt: new Date().toISOString(),
      actor: 'ai-service',
      trustLevel: 'safe-act',
      action: 'bridge.invoke',
      classification: {
        id: parsedRequest.capabilityId,
        category: 'integration',
        impact: risky ? 'high' : 'medium',
        reversible: !risky,
        requiresNetwork: true,
        description: `Legacy bridge invocation for ${parsedRequest.capabilityId}`,
      },
      audit: {
        traceId: trace.traceId ?? requestId,
        correlationId: trace.correlationId,
        requestId: trace.requestId,
        source: 'apps.ai',
        timestamp: new Date().toISOString(),
      },
      context: {
        capabilityId: parsedRequest.capabilityId,
      },
    });

    if (approval.result.outcome === 'deny') {
      return res.status(403).json({
        error: 'bridge invocation denied by approval policy',
        approval: approval.result,
      });
    }

    if (approval.result.outcome === 'require_approval') {
      return res.status(409).json({
        error: 'bridge invocation requires operator approval',
        approval: approval.result,
      });
    }
  }

  logger.info(
    'bridge.invoke.requested',
    'legacy bridge invoke requested',
    {
      requestId,
      capabilityId: parsedRequest.capabilityId,
      mode: cfg.BRIDGE_MODE,
      bridgeEnabled: cfg.BRIDGE_ENABLED,
    },
    trace,
  );

  const data = bridge.execute(parsedRequest);

  logger.info(
    'bridge.invoke.completed',
    'legacy bridge invoke completed',
    {
      requestId,
      capabilityId: parsedRequest.capabilityId,
      attempted: data.attempted,
      bridgeEnabled: data.bridgeEnabled,
      mode: data.mode,
      status: data.response?.status,
    },
    trace,
  );

  if (!data.attempted) return res.status(503).json({ data });
  if (data.response?.status === 'error') return res.status(502).json({ data });
  return res.status(200).json({ data });
});

app.post('/ai/execute', (req: Request, res: Response) => {
  const trace = getTelemetryContext(req);

  const aiRequest = AiServiceRequestSchema.parse({
    requestId: typeof req.body.requestId === 'string' ? req.body.requestId : randomUUID(),
    capabilityId: typeof req.body.capabilityId === 'string' ? req.body.capabilityId : undefined,
    input:
      req.body.input && typeof req.body.input === 'object'
        ? (req.body.input as Record<string, unknown>)
        : {},
    target: typeof req.body.target === 'string' ? req.body.target : undefined,
    trace: {
      traceId: trace.traceId,
      correlationId: trace.correlationId,
      requestId: trace.requestId,
    },
  });

  if (shouldUseLegacyBridge(aiRequest.capabilityId, aiRequest.target)) {
    const bridgeResult = bridge.execute({
      requestId: aiRequest.requestId,
      operation: 'invoke',
      capabilityId: aiRequest.capabilityId,
      payload: aiRequest.input,
      context: {
        traceId: aiRequest.trace.traceId,
        correlationId: aiRequest.trace.correlationId,
        requestId: aiRequest.trace.requestId,
        actor: 'ai-service',
        source: 'apps.ai',
      },
    });

    const response = AiServiceResponseSchema.parse({
      requestId: aiRequest.requestId,
      capabilityId: aiRequest.capabilityId,
      target: 'legacy_python_runtime',
      status:
        !bridgeResult.attempted || bridgeResult.response?.status === 'unavailable'
          ? 'unavailable'
          : bridgeResult.response?.status === 'error'
            ? 'failed'
            : 'succeeded',
      output: bridgeResult.response?.data ?? {},
      error:
        bridgeResult.response?.status === 'error'
          ? {
              code: bridgeResult.response.error?.code ?? 'BRIDGE_ERROR',
              message: bridgeResult.response.error?.message ?? 'bridge invocation failed',
              details: bridgeResult.response.error?.details,
            }
          : undefined,
    });

    const statusCode =
      response.status === 'succeeded' ? 200 : response.status === 'unavailable' ? 503 : 502;
    return res.status(statusCode).json({ data: response });
  }

  const result = skillRegistry.execute({
    skillId: aiRequest.capabilityId,
    skillName: aiRequest.capabilityId,
    input: aiRequest.input,
    context: {
      traceId: aiRequest.trace.traceId,
      actor: 'ai-service',
    },
  });

  const response = AiServiceResponseSchema.parse({
    requestId: aiRequest.requestId,
    capabilityId: aiRequest.capabilityId,
    target: 'typescript_service',
    status:
      result.status === 'succeeded'
        ? 'succeeded'
        : result.status === 'blocked'
          ? 'blocked'
          : 'failed',
    output: result.output,
    error: result.error,
  });

  const statusCode =
    response.status === 'succeeded' ? 200 : response.status === 'blocked' ? 409 : 422;
  return res.status(statusCode).json({ data: response });
});
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
