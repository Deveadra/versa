import { randomUUID } from 'node:crypto';
import {
  MemoryConsolidationRequestSchema,
  MemoryConsolidationResultSchema,
  MemoryReadRequestSchema,
  MemoryRecordSchema,
  MemoryWriteRequestSchema,
  type MemoryConsolidationRequest,
  type MemoryConsolidationResult,
  type MemoryReadRequest,
  type MemoryRecord,
  type MemoryWriteRequest,
} from '@versa/shared';

export type MemoryGateway = {
  write(input: MemoryWriteRequest): MemoryRecord;
  read(input?: MemoryReadRequest): MemoryRecord[];
  getById(memoryId: string): MemoryRecord | null;
  consolidate(input: MemoryConsolidationRequest): MemoryConsolidationResult;
};

export type MemoryRepository = {
  create(input: MemoryWriteRequest): MemoryRecord;
  list(input?: MemoryReadRequest): MemoryRecord[];
  getById(memoryId: string): MemoryRecord | null;
  consolidate(input: MemoryConsolidationRequest): MemoryRecord;
};

type MemoryGatewayDependencies = {
  now?: () => string;
  idFactory?: () => string;
};

const sortByRecency = (records: MemoryRecord[]) =>
  records.slice().sort((a, b) => b.updatedAt.localeCompare(a.updatedAt));

const normalizeText = (value: string) => value.trim().toLowerCase();

export const createMemoryGateway = (
  options?: {
    seed?: MemoryRecord[];
    repository?: MemoryRepository;
  } & MemoryGatewayDependencies,
): MemoryGateway => {
  const now = options?.now ?? (() => new Date().toISOString());
  const idFactory = options?.idFactory ?? (() => `mem_${randomUUID().slice(0, 8)}`);

  const store = new Map<string, MemoryRecord>();
  for (const item of options?.seed ?? []) {
    const parsed = MemoryRecordSchema.parse(item);
    store.set(parsed.id, parsed);
  }

  const write = (input: MemoryWriteRequest): MemoryRecord => {
    const parsed = MemoryWriteRequestSchema.parse(input);
    const timestamp = now();
    const record = MemoryRecordSchema.parse({
      id: idFactory(),
      tier: parsed.tier,
      summary: parsed.summary,
      content: parsed.content,
      metadata: parsed.metadata,
      createdAt: timestamp,
      updatedAt: timestamp,
      lastAccessedAt: timestamp,
    });

    store.set(record.id, record);
    return record;
  };

  const getById = (memoryId: string): MemoryRecord | null => {
    const current = store.get(memoryId);
    if (!current) return null;

    const touched = MemoryRecordSchema.parse({
      ...current,
      lastAccessedAt: now(),
      updatedAt: current.updatedAt,
    });
    store.set(memoryId, touched);
    return touched;
  };

  const read = (input?: MemoryReadRequest): MemoryRecord[] => {
    if (options?.repository) {
      return options.repository.list(input);
    }

    const parsed = MemoryReadRequestSchema.parse(input ?? {});
    const tiers = parsed.tiers ? new Set(parsed.tiers) : null;
    const q = parsed.text ? normalizeText(parsed.text) : null;

    const matches = Array.from(store.values()).filter((item) => {
      if (tiers && !tiers.has(item.tier)) return false;
      if (parsed.minConfidence !== undefined && item.metadata.confidence < parsed.minConfidence) {
        return false;
      }

      if (!q) return true;
      const summaryHit = normalizeText(item.summary).includes(q);
      const contentHit = normalizeText(JSON.stringify(item.content)).includes(q);
      const tagsHit = item.metadata.tags.some((tag: string) => normalizeText(tag).includes(q));
      return summaryHit || contentHit || tagsHit;
    });

    return sortByRecency(matches).slice(0, parsed.limit);
  };

  const consolidate = (input: MemoryConsolidationRequest): MemoryConsolidationResult => {
    const parsed = MemoryConsolidationRequestSchema.parse(input);

    if (options?.repository) {
      const promoted = options.repository.consolidate(parsed);
      return MemoryConsolidationResultSchema.parse({
        promotedMemory: promoted,
        linkedSourceCount: parsed.sourceMemoryIds.length,
      });
    }

    const missing = parsed.sourceMemoryIds.filter((id: string) => !store.has(id));
    if (missing.length > 0) {
      throw new Error(`cannot consolidate missing memories: ${missing.join(', ')}`);
    }

    const promoted = write({
      tier: parsed.targetTier,
      summary: parsed.summary,
      content: parsed.content,
      metadata: {
        ...parsed.metadata,
        provenance: {
          ...parsed.metadata.provenance,
          sourceMemoryIds: parsed.sourceMemoryIds,
          notes: parsed.reason,
        },
      },
    });

    return MemoryConsolidationResultSchema.parse({
      promotedMemory: promoted,
      linkedSourceCount: parsed.sourceMemoryIds.length,
    });
  };

  return {
    write: (input: MemoryWriteRequest): MemoryRecord => {
      if (options?.repository) {
        return options.repository.create(input);
      }
      return write(input);
    },
    read,
    getById: (memoryId: string): MemoryRecord | null => {
      if (options?.repository) {
        return options.repository.getById(memoryId);
      }
      return getById(memoryId);
    },
    consolidate,
  };
};

export type {
  MemoryConsolidationRequest,
  MemoryConsolidationResult,
  MemoryReadRequest,
  MemoryRecord,
  MemoryWriteRequest,
} from '@versa/shared';
