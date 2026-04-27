import { describe, expect, it } from 'vitest';
import { createEnvironmentGateway } from './index';

describe('createEnvironmentGateway', () => {
  it('creates an environment and returns context with records, relationships, access paths, and procedures', () => {
    let n = 0;
    const gateway = createEnvironmentGateway({
      now: () => '2026-01-01T00:00:00.000Z',
      idFactory: (prefix: string) => `${prefix}_00000${++n}`,
    });

    const environment = gateway.create({
      slug: 'prod/us-east-1',
      name: 'Production US East',
      metadata: {
        owner: 'platform',
        tags: ['production'],
        source: 'manual',
      },
    });

    const serviceRecord = gateway.upsertRecord(environment.slug, {
      id: 'enr_12345678',
      environmentId: environment.id,
      kind: 'service',
      name: 'core-api',
      attributes: {
        port: 4010,
      },
      metadata: {
        source: 'manual',
        tags: ['core'],
        confidence: 0.9,
        createdAt: '2026-01-01T00:00:00.000Z',
        updatedAt: '2026-01-01T00:00:00.000Z',
      },
    });

    const machineRecord = gateway.upsertRecord(environment.slug, {
      id: 'enr_87654321',
      environmentId: environment.id,
      kind: 'machine',
      name: 'core-host-01',
      attributes: {
        os: 'linux',
      },
      metadata: {
        source: 'manual',
        tags: ['host'],
        confidence: 0.9,
        createdAt: '2026-01-01T00:00:00.000Z',
        updatedAt: '2026-01-01T00:00:00.000Z',
      },
    });

    const relationship = gateway.addRelationship(environment.slug, {
      id: 'enl_12345678',
      environmentId: environment.id,
      fromEntityId: 'enr_12345678',
      toEntityId: 'enr_87654321',
      relation: 'runs_on',
      direction: 'directed',
      createdAt: '2026-01-01T00:00:00.000Z',
    });

    const accessPath = gateway.addAccessPath(environment.slug, {
      id: 'eap_12345678',
      environmentId: environment.id,
      entityId: 'enr_12345678',
      name: 'core health endpoint',
      method: 'http',
      endpoint: 'http://localhost:4010/health',
      prerequisites: ['core process running'],
      commandRefIds: [],
      createdAt: '2026-01-01T00:00:00.000Z',
    });

    const procedure = gateway.addProcedure(environment.slug, {
      id: 'epr_12345678',
      environmentId: environment.id,
      name: 'validate core health',
      intent: 'confirm core API availability',
      targetEntityIds: ['enr_12345678'],
      steps: [
        {
          order: 1,
          instruction: 'curl http://localhost:4010/health',
          expectedOutcome: '{"ok":true}',
        },
      ],
      createdAt: '2026-01-01T00:00:00.000Z',
      updatedAt: '2026-01-01T00:00:00.000Z',
      tags: ['healthcheck'],
    });

    const context = gateway.getContextBundle(environment.slug, 10);
    expect(serviceRecord?.kind).toBe('service');
    expect(machineRecord?.kind).toBe('machine');
    expect(relationship?.relation).toBe('runs_on');
    expect(accessPath?.method).toBe('http');
    expect(procedure?.steps[0]?.order).toBe(1);
    expect(context?.records).toHaveLength(2);
    expect(context?.relationships).toHaveLength(1);
    expect(context?.accessPaths).toHaveLength(1);
    expect(context?.procedures).toHaveLength(1);
  });
});
