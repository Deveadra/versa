import { describe, expect, it } from 'vitest';
import {
  createDoctrineStore,
  defaultDoctrine,
  DoctrineLoadError,
  getDoctrine,
  loadDoctrine,
  loadDoctrineFromFile,
  parseDoctrine,
  updateDoctrine,
} from './index';

describe('doctrine package', () => {
  it('loads default doctrine when no file path is provided', () => {
    const loaded = loadDoctrine();

    expect(loaded.source).toBe('default');
    expect(loaded.doctrine.doctrineId).toBe('aerith.ultron');
  });

  it('falls back to default doctrine when configured file path is missing', () => {
    const loaded = loadDoctrine({
      enabled: true,
      filePath: 'state/does-not-exist.json',
    });

    expect(loaded.source).toBe('default');
    expect(loaded.doctrine.doctrineId).toBe('aerith.ultron');
  });

  it('throws DoctrineLoadError with clear code when missing file fallback is disabled', () => {
    expect(() =>
      loadDoctrine({
        enabled: true,
        filePath: 'state/does-not-exist.json',
        fallbackOnFileNotFound: false,
      }),
    ).toThrowError(DoctrineLoadError);

    try {
      loadDoctrine({
        enabled: true,
        filePath: 'state/does-not-exist.json',
        fallbackOnFileNotFound: false,
      });
    } catch (error) {
      expect(error).toBeInstanceOf(DoctrineLoadError);
      expect((error as DoctrineLoadError).code).toBe('DOCTRINE_FILE_NOT_FOUND');
    }
  });

  it('throws DoctrineLoadError when doctrine file json is invalid', () => {
    expect(() => loadDoctrineFromFile('package.json')).toThrowError(DoctrineLoadError);

    try {
      loadDoctrineFromFile('package.json');
    } catch (error) {
      expect(error).toBeInstanceOf(DoctrineLoadError);
      expect((error as DoctrineLoadError).code).toBe('DOCTRINE_SCHEMA_INVALID');
    }
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

  it('supports bounded history to avoid unbounded growth', () => {
    const store = createDoctrineStore({ maxHistory: 2 });

    const v101 = updateDoctrine(store, {
      ...defaultDoctrine,
      version: '1.0.1',
      metadata: {
        ...defaultDoctrine.metadata,
        updatedAt: '2026-01-02T00:00:00.000Z',
        changeSummary: 'v1.0.1',
      },
    });

    const v102 = updateDoctrine(v101, {
      ...defaultDoctrine,
      version: '1.0.2',
      metadata: {
        ...defaultDoctrine.metadata,
        updatedAt: '2026-01-03T00:00:00.000Z',
        changeSummary: 'v1.0.2',
      },
    });

    expect(v102.history).toHaveLength(2);
    expect(v102.history[0]?.version).toBe('1.0.1');
    expect(v102.history[1]?.version).toBe('1.0.2');
  });
});
