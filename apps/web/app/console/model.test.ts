import { describe, expect, it } from 'vitest';
import { deriveApprovalVisibilitySnapshot } from './model';

describe('deriveApprovalVisibilitySnapshot', () => {
  it('derives approval visibility counts from skills and events', () => {
    const snapshot = deriveApprovalVisibilitySnapshot(
      [
        {
          id: 'skill_1',
          name: 'safe_skill',
          metadata: {
            description: 'safe skill',
            version: '0.1.0',
            inputs: [],
            outputs: [],
            requiredTools: [],
            requiredResources: [],
            validationChecks: [],
            failureHandling: { retryable: false, maxRetries: 0 },
            approval: { required: true },
          },
        },
        {
          id: 'skill_2',
          name: 'ungoverned_skill',
          metadata: {
            description: 'ungoverned skill',
            version: '0.1.0',
            inputs: [],
            outputs: [],
            requiredTools: [],
            requiredResources: [],
            validationChecks: [],
            failureHandling: { retryable: false, maxRetries: 0 },
            approval: { required: false },
          },
        },
      ],
      [
        {
          eventId: 'evt_12345678',
          eventType: 'task.updated',
          actor: 'ai-service',
          timestamp: new Date().toISOString(),
          domain: 'system',
          entityRef: { type: 'policy', id: 'plc_12345678' },
          payload: { requiresApproval: true },
          sensitivity: 'internal',
          traceId: 'trace-1',
        },
        {
          eventId: 'evt_87654321',
          eventType: 'task.created',
          actor: 'core-api',
          timestamp: new Date().toISOString(),
          domain: 'core',
          entityRef: { type: 'task', id: 'tsk_12345678' },
          payload: {},
          sensitivity: 'internal',
          traceId: 'trace-2',
        },
      ],
    );

    expect(snapshot.governedSkillCount).toBe(1);
    expect(snapshot.approvalRelatedEventCount).toBe(1);
  });
});
