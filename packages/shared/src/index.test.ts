import { describe, expect, it } from 'vitest';
import {
  DoctrineSchema,
  DomainEventSchema,
  MemoryConsolidationRequestSchema,
  MemoryReadRequestSchema,
  MemoryWriteRequestSchema,
  TelemetryEventSchema,
  WorkspaceCheckpointCreateRequestSchema,
  WorkspaceCreateRequestSchema,
  WorkspaceStatePatchSchema,
} from './index';

describe('DomainEventSchema', () => {
  it('validates a task.created event', () => {
    const parsed = DomainEventSchema.parse({
      eventId: 'evt_12345678',
      eventType: 'task.created',
      actor: 'system',
      timestamp: new Date().toISOString(),
      domain: 'core',
      entityRef: { type: 'task', id: 'tsk_12345678' },
      payload: { title: 'hello' },
      sensitivity: 'internal',
      traceId: 'trace-1',
    });

    expect(parsed.eventType).toBe('task.created');
  });
});

describe('TelemetryEventSchema', () => {
  it('validates a telemetry event with trace conventions', () => {
    const parsed = TelemetryEventSchema.parse({
      eventId: 'evt_87654321',
      eventType: 'http.request.completed',
      level: 'info',
      message: 'request completed',
      timestamp: new Date().toISOString(),
      actor: {
        service: 'core',
        source: 'http',
      },
      context: {
        traceId: 'trace-123',
        correlationId: 'corr-123',
        runId: 'run-123',
        requestId: 'req-123',
      },
      attributes: {
        method: 'GET',
        route: '/health',
        statusCode: 200,
      },
    });

    expect(parsed.context.traceId).toBe('trace-123');
    expect(parsed.level).toBe('info');
  });
});

describe('DoctrineSchema', () => {
  it('validates a doctrine document', () => {
    const parsed = DoctrineSchema.parse({
      doctrineId: 'aerith.ultron',
      version: '1.0.0',
      mission: 'Protect operator intent while executing reliable outcomes.',
      operatorPrinciples: ['Be truthful', 'Prefer reversible actions'],
      responseStyle: {
        tone: 'direct',
        verbosity: 'concise',
        markdownRequired: true,
        citationStyle: 'repo-link',
        forbiddenPhrases: ['Great', 'Certainly'],
      },
      decisionPriorities: ['operator_safety', 'mission_alignment', 'truthfulness'],
      escalationRules: [
        {
          id: 'approval-destructive',
          condition: 'destructive operation is requested',
          severity: 'high',
          action: 'request explicit operator approval',
        },
      ],
      autonomyBoundaries: [
        {
          action: 'git push',
          requiresApproval: true,
          rationale: 'remote side effects require operator confirmation',
        },
      ],
      safetyNoGoActions: [
        {
          id: 'no-unauthorized-destructive-changes',
          rule: 'Do not execute destructive repository actions without explicit authorization.',
          rationale: 'Preserve operator control and recoverability.',
        },
      ],
      ownership: {
        team: 'platform',
        maintainers: ['@deveadra'],
      },
      metadata: {
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
        changeSummary: 'initial doctrine baseline',
      },
    });

    expect(parsed.doctrineId).toBe('aerith.ultron');
    expect(parsed.decisionPriorities[0]).toBe('operator_safety');
  });
});

describe('Memory contracts', () => {
  it('validates memory write/read/consolidation contracts', () => {
    const write = MemoryWriteRequestSchema.parse({
      tier: 'episodic',
      summary: 'User completed afternoon workout',
      content: { durationMinutes: 45, type: 'cardio' },
      metadata: {
        confidence: 0.82,
        source: 'manual',
        sensitivity: 'private',
        retention: {
          strategy: 'durable',
          ttlDays: 365,
          decayRate: 0.15,
        },
        provenance: {
          actor: 'core-api',
          traceId: 'trace-memory-1',
          subsystem: 'core',
        },
        tags: ['health', 'habit'],
      },
    });

    const read = MemoryReadRequestSchema.parse({
      text: 'workout',
      tiers: ['episodic', 'semantic'],
      minConfidence: 0.5,
      limit: 10,
    });

    const consolidation = MemoryConsolidationRequestSchema.parse({
      sourceMemoryIds: ['mem_12345678', 'mem_87654321'],
      targetTier: 'semantic',
      summary: 'User tends to exercise in afternoons',
      reason: 'pattern observed over multiple sessions',
      content: { cadence: 'weekly' },
      metadata: write.metadata,
    });

    expect(write.tier).toBe('episodic');
    expect(read.limit).toBe(10);
    expect(consolidation.targetTier).toBe('semantic');
  });
});

describe('Workspace contracts', () => {
  it('validates workspace create/state/checkpoint contracts', () => {
    const create = WorkspaceCreateRequestSchema.parse({
      slug: 'deveadra/versa',
      name: 'Versa',
      repository: 'github.com/Deveadra/versa',
      metadata: {
        owner: 'platform',
        tags: ['redesign', 'ws05'],
        source: 'manual',
      },
      state: {
        currentObjective: 'Ship WS05 workspace-state foundations',
        activeBlockers: [{ description: 'None currently', status: 'mitigated' }],
        recentDecisions: [
          {
            summary: 'Use explicit workspace records and checkpoints',
            decidedAt: new Date().toISOString(),
          },
        ],
        importantFiles: [{ path: 'apps/core/src/server.ts', reason: 'workspace API endpoints' }],
        knownCommands: [{ command: 'pnpm test', description: 'full workspace tests' }],
        validatedProcedures: [{ name: 'monorepo validation', command: 'pnpm lint' }],
        nextRecommendedActions: [{ action: 'Add workspace API read path', priority: 'high' }],
      },
    });

    const patch = WorkspaceStatePatchSchema.parse({
      nextRecommendedActions: [
        {
          action: 'Run WS05 validation suite',
          priority: 'high',
        },
      ],
    });

    const checkpoint = WorkspaceCheckpointCreateRequestSchema.parse({
      summary: 'Validated workspace foundations',
      createdBy: 'orion',
    });

    expect(create.slug).toBe('deveadra/versa');
    expect(patch.nextRecommendedActions?.[0]?.priority).toBe('high');
    expect(checkpoint.createdBy).toBe('orion');
  });
});
