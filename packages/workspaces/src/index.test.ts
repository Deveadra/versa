import { describe, expect, it } from 'vitest';
import { createWorkspaceGateway } from './index';

describe('createWorkspaceGateway', () => {
  it('creates and retrieves a workspace by slug', () => {
    const gateway = createWorkspaceGateway({
      now: () => '2026-04-30T00:00:00.000Z',
      idFactory: (prefix: string) => `${prefix}_00000001`,
    });

    const created = gateway.create({
      slug: 'deveadra/versa',
      name: 'Versa',
      repository: 'github.com/Deveadra/versa',
      metadata: { owner: 'platform', tags: ['ws24'], source: 'manual' },
      state: {
        currentObjective: 'WS24 coverage confidence',
        activeBlockers: [{ description: 'none', status: 'mitigated' }],
        recentDecisions: [{ summary: 'enforce explicit thresholds', decidedAt: '2026-04-30T00:00:00.000Z' }],
        importantFiles: [{ path: '.github/workflows/ci.yml', reason: 'coverage workflow' }],
        knownCommands: [{ command: 'pnpm test:coverage:ts', description: 'TS coverage + thresholds' }],
        validatedProcedures: [{ name: 'ci-quality', command: 'pnpm lint && pnpm typecheck && pnpm test' }],
        nextRecommendedActions: [{ action: 'review coverage artifacts', priority: 'high' }],
      },
    });

    const found = gateway.getBySlug('deveadra/versa');
    const bundle = gateway.getContextBundle('deveadra/versa');

    expect(created.slug).toBe('deveadra/versa');
    expect(found?.id).toBe(created.id);
    expect(bundle?.summary.currentObjective).toBe('WS24 coverage confidence');
  });
});
