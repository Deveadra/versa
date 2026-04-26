import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { DoctrineSchema, type Doctrine } from '@versa/shared';
import { defaultDoctrine } from './default-doctrine';

export type DoctrineLoadSource = 'default' | 'file';

export type DoctrineLoadResult = {
  doctrine: Doctrine;
  source: DoctrineLoadSource;
  resolvedPath?: string;
};

export class DoctrineLoadError extends Error {
  readonly code:
    | 'DOCTRINE_FILE_NOT_FOUND'
    | 'DOCTRINE_FILE_READ_FAILED'
    | 'DOCTRINE_JSON_INVALID'
    | 'DOCTRINE_SCHEMA_INVALID';

  readonly resolvedPath?: string;

  constructor(
    code:
      | 'DOCTRINE_FILE_NOT_FOUND'
      | 'DOCTRINE_FILE_READ_FAILED'
      | 'DOCTRINE_JSON_INVALID'
      | 'DOCTRINE_SCHEMA_INVALID',
    message: string,
    resolvedPath?: string,
  ) {
    super(message);
    this.name = 'DoctrineLoadError';
    this.code = code;
    this.resolvedPath = resolvedPath;
  }
}

export type DoctrineStore = {
  current: Doctrine;
  history: Doctrine[];
  maxHistory?: number;
};

export const parseDoctrine = (input: unknown): Doctrine => DoctrineSchema.parse(input);

export const loadDoctrineFromFile = (path: string): DoctrineLoadResult => {
  const resolvedPath = resolve(path);
  let raw: string;

  try {
    raw = readFileSync(resolvedPath, 'utf-8');
  } catch (error) {
    const errorCode = (error as NodeJS.ErrnoException).code;
    if (errorCode === 'ENOENT') {
      throw new DoctrineLoadError(
        'DOCTRINE_FILE_NOT_FOUND',
        `Doctrine file not found at ${resolvedPath}`,
        resolvedPath,
      );
    }

    throw new DoctrineLoadError(
      'DOCTRINE_FILE_READ_FAILED',
      `Failed to read doctrine file at ${resolvedPath}: ${(error as Error).message}`,
      resolvedPath,
    );
  }

  let parsedJson: unknown;
  try {
    parsedJson = JSON.parse(raw);
  } catch (error) {
    throw new DoctrineLoadError(
      'DOCTRINE_JSON_INVALID',
      `Doctrine file at ${resolvedPath} contains invalid JSON: ${(error as Error).message}`,
      resolvedPath,
    );
  }

  let parsed: Doctrine;
  try {
    parsed = parseDoctrine(parsedJson);
  } catch (error) {
    throw new DoctrineLoadError(
      'DOCTRINE_SCHEMA_INVALID',
      `Doctrine file at ${resolvedPath} failed schema validation: ${(error as Error).message}`,
      resolvedPath,
    );
  }

  return {
    doctrine: parsed,
    source: 'file',
    resolvedPath,
  };
};

export const loadDoctrine = (options?: {
  filePath?: string | null;
  enabled?: boolean;
  fallback?: Doctrine;
  fallbackOnFileNotFound?: boolean;
}): DoctrineLoadResult => {
  const enabled = options?.enabled ?? true;
  const fallback = options?.fallback ?? defaultDoctrine;
  const fallbackOnFileNotFound = options?.fallbackOnFileNotFound ?? true;

  if (!enabled) {
    return {
      doctrine: parseDoctrine(fallback),
      source: 'default',
    };
  }

  const filePath = options?.filePath?.trim();
  if (!filePath) {
    return {
      doctrine: parseDoctrine(fallback),
      source: 'default',
    };
  }

  try {
    return loadDoctrineFromFile(filePath);
  } catch (error) {
    if (
      fallbackOnFileNotFound &&
      error instanceof DoctrineLoadError &&
      error.code === 'DOCTRINE_FILE_NOT_FOUND'
    ) {
      return {
        doctrine: parseDoctrine(fallback),
        source: 'default',
      };
    }

    throw error;
  }
};

export const createDoctrineStore = (options?: { seed?: Doctrine; maxHistory?: number }): DoctrineStore => {
  const maxHistory = options?.maxHistory;
  if (maxHistory !== undefined && maxHistory < 1) {
    throw new Error('maxHistory must be at least 1 when provided');
  }

  const seed = options?.seed;
  const initial = parseDoctrine(seed ?? defaultDoctrine);

  return {
    current: initial,
    history: [initial],
    maxHistory,
  };
};

export const updateDoctrine = (store: DoctrineStore, next: Doctrine): DoctrineStore => {
  const parsed = parseDoctrine(next);

  const nextHistory = [...store.history, parsed];
  const history =
    store.maxHistory && nextHistory.length > store.maxHistory
      ? nextHistory.slice(nextHistory.length - store.maxHistory)
      : nextHistory;

  return {
    current: parsed,
    history,
    maxHistory: store.maxHistory,
  };
};

export const getDoctrine = (store: DoctrineStore): Doctrine => store.current;

export { defaultDoctrine };
export type { Doctrine } from '@versa/shared';
