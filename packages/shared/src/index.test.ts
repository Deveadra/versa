import { describe, expect, it } from 'vitest';
import { DomainEventSchema } from './index';

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
