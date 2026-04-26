import { describe, expect, it } from 'vitest';
import {
  createDoctrineStore,
  defaultDoctrine,
  getDoctrine,
  loadDoctrine,
  parseDoctrine,
  updateDoctrine,
} from './index';

describe('doctrine package', () => {
  it('loads default doctrine when no file path is provided', () => {
    const loaded = loadDoctrine();

    expect(loaded.source).toBe('default');
    expect(loaded.doctrine.doctrineId).toBe('aerith.ultron');
  });

  it('validates doctrine payloads through shared schema', () => {
    const parsed = parseDoctrine(defaultDoctrine);

    expect(parsed.version).toBe('1.0.0');
    expect(parsed.operatorPrinciples.length).toBeGreaterThan(0);
  });

  it('tracks doctrine history through the store helper', () => {
    const store = createDoctrineStore();
    const next = {
      ...defaultDoctrine,
      version: '1.0.1',
      metadata: {
        ...defaultDoctrine.metadata,
        updatedAt: new Date().toISOString(),
        changeSummary: 'increment version for test',
      },
    };

    const updated = updateDoctrine(store, next);

    expect(getDoctrine(updated).version).toBe('1.0.1');
    expect(updated.history).toHaveLength(2);
  });
});
