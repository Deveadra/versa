import { describe, expect, it, vi } from 'vitest';
import { createLogger, createRequestTelemetryMiddleware } from './index';

describe('createLogger', () => {
  it('emits structured telemetry with trace context', () => {
    const sink = vi.fn();
    const logger = createLogger({
      actor: { service: 'core', source: 'http' },
      sink,
      context: { runId: 'run-1' },
    });

    logger.info('app.started', 'core started', { port: 4000 }, { traceId: 'trace-1' });

    expect(sink).toHaveBeenCalledTimes(1);
    const event = sink.mock.calls[0]?.[0];
    expect(event.eventType).toBe('app.started');
    expect(event.context.traceId).toBe('trace-1');
    expect(event.context.runId).toBe('run-1');
    expect(event.attributes.port).toBe(4000);
  });
});

describe('createRequestTelemetryMiddleware', () => {
  it('emits request start and completion events', () => {
    const sink = vi.fn();
    const logger = createLogger({
      actor: { service: 'ai', source: 'http' },
      sink,
    });

    const middleware = createRequestTelemetryMiddleware(logger);

    let finishListener: (() => void) | undefined;
    const req = {
      method: 'GET',
      originalUrl: '/health',
      headers: { 'x-trace-id': 'trace-2', 'x-request-id': 'req-2' },
    };
    const res = {
      statusCode: 200,
      on: (_event: 'finish', listener: () => void) => {
        finishListener = listener;
      },
    };

    middleware(req, res, () => undefined);
    finishListener?.();

    expect(sink).toHaveBeenCalledTimes(2);
    expect(sink.mock.calls[0]?.[0].eventType).toBe('http.request.started');
    expect(sink.mock.calls[1]?.[0].eventType).toBe('http.request.completed');
    expect((req as { telemetryContext?: { traceId?: string } }).telemetryContext?.traceId).toBe(
      'trace-2',
    );
  });
});
