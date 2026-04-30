import { describe, expect, it } from 'vitest';
import { createWorkspaceGateway } from './index';

const fixtureState = {
  currentObjective: 'Stabilize WS23 confidence suite',
  activeBlockers: [{ description: 'Need stronger workspace tests', status: 'active' as const }],
  recentDecisions: [
    {
      summary: 'Treat workspace state as durable project context',
      decidedAt: '2026-04-30T00:00:00.000Z',
    },
  ],
  importantFiles: [{ path: 'packages/workspaces/src/index.ts', reason: 'gateway behavior surface' }],
  knownCommands: [{ command: 'pnpm test', description: 'Run full repository tests' }],
  validatedProcedures: [{ name: 'quality suite', command: 'pnpm lint && pnpm typecheck && pnpm test' }],
  nextRecommendedActions: [{ action: 'Add durable-state edge-case tests', priority: 'high' as const }],
};

const createDeterministicGateway = () => {
  const timestamps = [
    '2026-04-30T00:00:00.000Z',
    '2026-04-30T00:01:00.000Z',
    '2026-04-30T00:02:00.000Z',
    '2026-04-30T00:03:00.000Z',
    '2026-04-30T00:04:00.000Z',
    '2026-04-30T00:05:00.000Z',
    '2026-04-30T00:06:00.000Z',
    '2026-04-30T00:07:00.000Z',
    '2026-04-30T00:08:00.000Z',
    '2026-04-30T00:09:00.000Z',
  ];
  let timeIndex = 0;
  let idIndex = 0;
  return createWorkspaceGateway({
    now: () => timestamps[Math.min(timeIndex++, timestamps.length - 1)] ?? timestamps[timestamps.length - 1]!,
    idFactory: (prefix: string) => `${prefix}_${String(++idIndex).padStart(8, '0')}`,
  });
};

describe('createWorkspaceGateway', () => {
  it('creates, lists, and resolves workspace by slug', () => {
    const gateway = createDeterministicGateway();

    const created = gateway.create({
      slug: 'deveadra/versa',
      name: 'Versa',
      repository: 'github.com/Deveadra/versa',
      metadata: {
        owner: 'platform',
        tags: ['ws23'],
        source: 'manual',
      },
      state: fixtureState,
    });

    const listed = gateway.list();
    const lookedUp = gateway.getBySlug('deveadra/versa');

    expect(created.id).toBe('wrk_00000001');
    expect(listed).toHaveLength(1);
    expect(listed[0]?.activeBlockerCount).toBe(1);
    expect(lookedUp?.slug).toBe('deveadra/versa');
  });

  it('updates state with repeated patches and keeps latest values', () => {
    const gateway = createDeterministicGateway();
    gateway.create({
      slug: 'deveadra/versa',
      name: 'Versa',
      repository: 'github.com/Deveadra/versa',
      metadata: { owner: 'platform', tags: [], source: 'manual' },
      state: fixtureState,
    });

    const first = gateway.updateState('deveadra/versa', {
      currentObjective: 'Draft workspace confidence test matrix',
      nextRecommendedActions: [{ action: 'Cover checkpoint ordering', priority: 'medium' }],
    });
    const second = gateway.updateState('deveadra/versa', {
      currentObjective: 'Finalize workspace confidence test matrix',
      nextRecommendedActions: [{ action: 'Cover missing lookup behavior', priority: 'high' }],
    });

    expect(first?.state.currentObjective).toBe('Draft workspace confidence test matrix');
    expect(second?.state.currentObjective).toBe('Finalize workspace confidence test matrix');
    expect(second?.state.nextRecommendedActions[0]?.priority).toBe('high');
    expect(second?.state.updatedAt).toBe('2026-04-30T00:02:00.000Z');
  });

  it('activates a workspace and updates lastActivatedAt', () => {
    const gateway = createDeterministicGateway();
    gateway.create({
      slug: 'deveadra/versa',
      name: 'Versa',
      repository: 'github.com/Deveadra/versa',
      metadata: { owner: 'platform', tags: [], source: 'manual' },
      state: fixtureState,
    });

    const activated = gateway.activate('deveadra/versa');

    expect(activated?.metadata.lastActivatedAt).toBe('2026-04-30T00:01:00.000Z');
    expect(activated?.metadata.updatedAt).toBe('2026-04-30T00:01:00.000Z');
  });

  it('creates checkpoints in reverse-chronological order and normalizes list limits', () => {
    const gateway = createDeterministicGateway();
    gateway.create({
      slug: 'deveadra/versa',
      name: 'Versa',
      repository: 'github.com/Deveadra/versa',
      metadata: { owner: 'platform', tags: [], source: 'manual' },
      state: fixtureState,
    });

    const first = gateway.checkpoint('deveadra/versa', {
      summary: 'Checkpoint one',
      createdBy: 'orion',
    });
    const second = gateway.checkpoint('deveadra/versa', {
      summary: 'Checkpoint two',
      createdBy: 'orion',
      snapshot: {
        ...fixtureState,
        currentObjective: 'Checkpointed custom snapshot',
      },
    });

    const defaultBundle = gateway.getContextBundle('deveadra/versa');
    const zeroLimitBundle = gateway.getContextBundle('deveadra/versa', 0);

    expect(first?.id).toBe('wcp_00000002');
    expect(second?.id).toBe('wcp_00000003');
    expect(defaultBundle?.recentCheckpoints[0]?.summary).toBe('Checkpoint two');
    expect(defaultBundle?.recentCheckpoints[1]?.summary).toBe('Checkpoint one');
    expect(zeroLimitBundle?.recentCheckpoints).toHaveLength(1);
    expect(zeroLimitBundle?.recentCheckpoints[0]?.summary).toBe('Checkpoint two');
  });

  it('returns null for missing workspace lookups and duplicate slug creation throws', () => {
    const gateway = createDeterministicGateway();

    gateway.create({
      slug: 'deveadra/versa',
      name: 'Versa',
      repository: 'github.com/Deveadra/versa',
      metadata: { owner: 'platform', tags: [], source: 'manual' },
      state: fixtureState,
    });

    expect(gateway.getBySlug('missing/workspace')).toBeNull();
    expect(gateway.updateState('missing/workspace', { currentObjective: 'No-op' })).toBeNull();
    expect(gateway.activate('missing/workspace')).toBeNull();
    expect(
      gateway.checkpoint('missing/workspace', {
        summary: 'No-op checkpoint',
        createdBy: 'orion',
      }),
    ).toBeNull();
    expect(gateway.getContextBundle('missing/workspace')).toBeNull();

    expect(() =>
      gateway.create({
        slug: 'deveadra/versa',
        name: 'Versa duplicate',
        repository: 'github.com/Deveadra/versa',
        metadata: { owner: 'platform', tags: [], source: 'manual' },
        state: fixtureState,
      }),
    ).toThrow('workspace slug already exists: deveadra/versa');
  });
});
