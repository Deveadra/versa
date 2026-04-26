import { describe, expect, it } from 'vitest';
import { DoctrineSchema, DomainEventSchema, TelemetryEventSchema } from './index';

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
