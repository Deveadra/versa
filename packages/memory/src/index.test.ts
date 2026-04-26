import { describe, expect, it } from 'vitest';
import { createMemoryGateway } from './index';

const baseMetadata = {
  confidence: 0.9,
  source: 'ai' as const,
  sensitivity: 'internal' as const,
  retention: {
    strategy: 'durable' as const,
    ttlDays: 180,
    decayRate: 0.1,
  },
  provenance: {
    actor: 'core-api',
    traceId: 'trace-1',
    subsystem: 'core',
  },
  tags: ['memory', 'test'],
};

describe('createMemoryGateway', () => {
  it('writes and reads memory records through one governed path', () => {
    const gateway = createMemoryGateway({
      now: () => '2026-01-01T00:00:00.000Z',
      idFactory: () => 'mem_fixed_1',
    });

    const created = gateway.write({
      tier: 'episodic',
      summary: 'User completed a high-focus study block',
      content: { minutes: 90, topic: 'physics' },
      metadata: baseMetadata,
    });

    expect(created.id).toBe('mem_fixed_1');
    expect(created.tier).toBe('episodic');
    expect(gateway.read({ text: 'study', tiers: ['episodic'], limit: 20 })).toHaveLength(1);
  });

  it('consolidates source memories into durable semantic/procedural records', () => {
    let n = 0;
    const gateway = createMemoryGateway({
      now: () => '2026-01-01T00:00:00.000Z',
      idFactory: () => `mem_00000${++n}`,
    });

    const first = gateway.write({
      tier: 'episodic',
      summary: 'User studies in the morning',
      content: { at: '08:00' },
      metadata: baseMetadata,
    });

    const second = gateway.write({
      tier: 'episodic',
      summary: 'User studies in the morning again',
      content: { at: '08:30' },
      metadata: baseMetadata,
    });

    const consolidated = gateway.consolidate({
      sourceMemoryIds: [first.id, second.id],
      targetTier: 'semantic',
      summary: 'User has a stable morning study habit',
      reason: 'repeated pattern detected',
      content: { pattern: 'morning-study' },
      metadata: baseMetadata,
    });

    expect(consolidated.linkedSourceCount).toBe(2);
    expect(consolidated.promotedMemory.tier).toBe('semantic');
    expect(consolidated.promotedMemory.metadata.provenance.sourceMemoryIds).toEqual([
      first.id,
      second.id,
    ]);
  });
});
