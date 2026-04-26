import { describe, expect, it } from 'vitest';
import { DomainEventSchema, TelemetryEventSchema } from './index';

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
