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

export type DoctrineStore = {
  current: Doctrine;
  history: Doctrine[];
};

export const parseDoctrine = (input: unknown): Doctrine => DoctrineSchema.parse(input);

export const loadDoctrineFromFile = (path: string): DoctrineLoadResult => {
  const resolvedPath = resolve(path);
  const raw = readFileSync(resolvedPath, 'utf-8');
  const parsed = parseDoctrine(JSON.parse(raw));

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
}): DoctrineLoadResult => {
  const enabled = options?.enabled ?? true;
  const fallback = options?.fallback ?? defaultDoctrine;

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

  return loadDoctrineFromFile(filePath);
};

export const createDoctrineStore = (seed?: Doctrine): DoctrineStore => {
  const initial = parseDoctrine(seed ?? defaultDoctrine);
  return {
    current: initial,
    history: [initial],
  };
};

export const updateDoctrine = (store: DoctrineStore, next: Doctrine): DoctrineStore => {
  const parsed = parseDoctrine(next);
  return {
    current: parsed,
    history: [...store.history, parsed],
  };
};

export const getDoctrine = (store: DoctrineStore): Doctrine => store.current;

export { defaultDoctrine };
export type { Doctrine } from '@versa/shared';
