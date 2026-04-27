import { randomUUID } from 'node:crypto';
import {
  EnvironmentAccessPathSchema,
  EnvironmentContextBundleSchema,
  EnvironmentProcedureSchema,
  EnvironmentRecordSchema,
  EnvironmentRelationshipSchema,
  EnvironmentSummarySchema,
  EnvironmentTwinCreateRequestSchema,
  EnvironmentTwinRecordSchema,
  normalizeEnvironmentItemLimit,
  type EnvironmentAccessPath,
  type EnvironmentContextBundle,
  type EnvironmentProcedure,
  type EnvironmentRecord,
  type EnvironmentRelationship,
  type EnvironmentSummary,
  type EnvironmentTwinCreateRequest,
  type EnvironmentTwinRecord,
} from '@versa/shared';

export type EnvironmentRepository = {
  create(input: EnvironmentTwinCreateRequest): EnvironmentTwinRecord;
  list(): EnvironmentSummary[];
  getBySlug(slug: string): EnvironmentTwinRecord | null;
  upsertRecord(environmentId: string, record: EnvironmentRecord): EnvironmentRecord | null;
  listRecords(environmentId: string, limit?: number): EnvironmentRecord[];
  addRelationship(environmentId: string, input: EnvironmentRelationship): EnvironmentRelationship | null;
  listRelationships(environmentId: string, limit?: number): EnvironmentRelationship[];
  addAccessPath(environmentId: string, input: EnvironmentAccessPath): EnvironmentAccessPath | null;
  listAccessPaths(environmentId: string, limit?: number): EnvironmentAccessPath[];
  addProcedure(environmentId: string, input: EnvironmentProcedure): EnvironmentProcedure | null;
  listProcedures(environmentId: string, limit?: number): EnvironmentProcedure[];
};

export type EnvironmentGateway = {
  create(input: EnvironmentTwinCreateRequest): EnvironmentTwinRecord;
  list(): EnvironmentSummary[];
  getBySlug(slug: string): EnvironmentTwinRecord | null;
  upsertRecord(slug: string, record: EnvironmentRecord): EnvironmentRecord | null;
  addRelationship(slug: string, input: EnvironmentRelationship): EnvironmentRelationship | null;
  addAccessPath(slug: string, input: EnvironmentAccessPath): EnvironmentAccessPath | null;
  addProcedure(slug: string, input: EnvironmentProcedure): EnvironmentProcedure | null;
  getContextBundle(slug: string, limit?: number): EnvironmentContextBundle | null;
};

type EnvironmentGatewayDependencies = {
  now?: () => string;
  idFactory?: (prefix: string) => string;
};

type InMemoryEnvironmentStore = {
  twinsById: Map<string, EnvironmentTwinRecord>;
  idBySlug: Map<string, string>;
  recordsByEnvironmentId: Map<string, Map<string, EnvironmentRecord>>;
  relationshipsByEnvironmentId: Map<string, EnvironmentRelationship[]>;
  accessPathsByEnvironmentId: Map<string, EnvironmentAccessPath[]>;
  proceduresByEnvironmentId: Map<string, EnvironmentProcedure[]>;
};

const createInMemoryStore = (): InMemoryEnvironmentStore => ({
  twinsById: new Map(),
  idBySlug: new Map(),
  recordsByEnvironmentId: new Map(),
  relationshipsByEnvironmentId: new Map(),
  accessPathsByEnvironmentId: new Map(),
  proceduresByEnvironmentId: new Map(),
});

const ensureStore = (store: InMemoryEnvironmentStore, environmentId: string) => {
  if (!store.recordsByEnvironmentId.has(environmentId)) {
    store.recordsByEnvironmentId.set(environmentId, new Map());
  }
  if (!store.relationshipsByEnvironmentId.has(environmentId)) {
    store.relationshipsByEnvironmentId.set(environmentId, []);
  }
  if (!store.accessPathsByEnvironmentId.has(environmentId)) {
    store.accessPathsByEnvironmentId.set(environmentId, []);
  }
  if (!store.proceduresByEnvironmentId.has(environmentId)) {
    store.proceduresByEnvironmentId.set(environmentId, []);
  }
};

const createInMemoryRepository = (
  deps: Required<EnvironmentGatewayDependencies>,
  store = createInMemoryStore(),
): EnvironmentRepository => ({
  create: (input: EnvironmentTwinCreateRequest): EnvironmentTwinRecord => {
    const parsed = EnvironmentTwinCreateRequestSchema.parse(input);
    if (store.idBySlug.has(parsed.slug)) {
      throw new Error(`environment slug already exists: ${parsed.slug}`);
    }

    const timestamp = deps.now();
    const twin = EnvironmentTwinRecordSchema.parse({
      id: deps.idFactory('env'),
      slug: parsed.slug,
      name: parsed.name,
      metadata: {
        owner: parsed.metadata.owner,
        tags: parsed.metadata.tags ?? [],
        source: parsed.metadata.source ?? 'manual',
        createdAt: timestamp,
        updatedAt: timestamp,
        lastValidatedAt: parsed.metadata.lastValidatedAt,
      },
    });

    store.twinsById.set(twin.id, twin);
    store.idBySlug.set(twin.slug, twin.id);
    ensureStore(store, twin.id);
    return twin;
  },

  list: (): EnvironmentSummary[] =>
    Array.from(store.twinsById.values())
      .map((twin) => {
        const records = Array.from(store.recordsByEnvironmentId.get(twin.id)?.values() ?? []);
        return EnvironmentSummarySchema.parse({
          id: twin.id,
          slug: twin.slug,
          name: twin.name,
          owner: twin.metadata.owner,
          recordCount: records.length,
          relationshipCount: (store.relationshipsByEnvironmentId.get(twin.id) ?? []).length,
          procedureCount: (store.proceduresByEnvironmentId.get(twin.id) ?? []).length,
          updatedAt: twin.metadata.updatedAt,
          lastValidatedAt: twin.metadata.lastValidatedAt,
        });
      })
      .sort((a, b) => b.updatedAt.localeCompare(a.updatedAt)),

  getBySlug: (slug: string): EnvironmentTwinRecord | null => {
    const id = store.idBySlug.get(slug);
    if (!id) return null;
    return store.twinsById.get(id) ?? null;
  },

  upsertRecord: (environmentId: string, record: EnvironmentRecord): EnvironmentRecord | null => {
    const twin = store.twinsById.get(environmentId);
    if (!twin) return null;

    ensureStore(store, environmentId);
    const parsed = EnvironmentRecordSchema.parse(record);
    const records = store.recordsByEnvironmentId.get(environmentId)!;
    records.set(parsed.id, parsed);

    const updatedTwin = EnvironmentTwinRecordSchema.parse({
      ...twin,
      metadata: {
        ...twin.metadata,
        updatedAt: deps.now(),
      },
    });
    store.twinsById.set(environmentId, updatedTwin);
    return parsed;
  },

  listRecords: (environmentId: string, limit = 50): EnvironmentRecord[] => {
    const parsedLimit = normalizeEnvironmentItemLimit(limit, 50, 200);
    const records = Array.from(store.recordsByEnvironmentId.get(environmentId)?.values() ?? []);
    return records
      .slice()
      .sort((a, b) => b.metadata.updatedAt.localeCompare(a.metadata.updatedAt))
      .slice(0, parsedLimit);
  },

  addRelationship: (
    environmentId: string,
    input: EnvironmentRelationship,
  ): EnvironmentRelationship | null => {
    const twin = store.twinsById.get(environmentId);
    if (!twin) return null;

    ensureStore(store, environmentId);
    const parsed = EnvironmentRelationshipSchema.parse(input);
    const list = store.relationshipsByEnvironmentId.get(environmentId)!;
    list.unshift(parsed);

    const updatedTwin = EnvironmentTwinRecordSchema.parse({
      ...twin,
      metadata: {
        ...twin.metadata,
        updatedAt: deps.now(),
      },
    });
    store.twinsById.set(environmentId, updatedTwin);
    return parsed;
  },

  listRelationships: (environmentId: string, limit = 50): EnvironmentRelationship[] => {
    const parsedLimit = normalizeEnvironmentItemLimit(limit, 50, 200);
    return (store.relationshipsByEnvironmentId.get(environmentId) ?? []).slice(0, parsedLimit);
  },

  addAccessPath: (environmentId: string, input: EnvironmentAccessPath): EnvironmentAccessPath | null => {
    const twin = store.twinsById.get(environmentId);
    if (!twin) return null;

    ensureStore(store, environmentId);
    const parsed = EnvironmentAccessPathSchema.parse(input);
    const list = store.accessPathsByEnvironmentId.get(environmentId)!;
    list.unshift(parsed);

    const updatedTwin = EnvironmentTwinRecordSchema.parse({
      ...twin,
      metadata: {
        ...twin.metadata,
        updatedAt: deps.now(),
      },
    });
    store.twinsById.set(environmentId, updatedTwin);
    return parsed;
  },

  listAccessPaths: (environmentId: string, limit = 50): EnvironmentAccessPath[] => {
    const parsedLimit = normalizeEnvironmentItemLimit(limit, 50, 200);
    return (store.accessPathsByEnvironmentId.get(environmentId) ?? []).slice(0, parsedLimit);
  },

  addProcedure: (environmentId: string, input: EnvironmentProcedure): EnvironmentProcedure | null => {
    const twin = store.twinsById.get(environmentId);
    if (!twin) return null;

    ensureStore(store, environmentId);
    const parsed = EnvironmentProcedureSchema.parse(input);
    const list = store.proceduresByEnvironmentId.get(environmentId)!;
    list.unshift(parsed);

    const updatedTwin = EnvironmentTwinRecordSchema.parse({
      ...twin,
      metadata: {
        ...twin.metadata,
        updatedAt: deps.now(),
      },
    });
    store.twinsById.set(environmentId, updatedTwin);
    return parsed;
  },

  listProcedures: (environmentId: string, limit = 50): EnvironmentProcedure[] => {
    const parsedLimit = normalizeEnvironmentItemLimit(limit, 50, 200);
    return (store.proceduresByEnvironmentId.get(environmentId) ?? []).slice(0, parsedLimit);
  },
});

export const createEnvironmentGateway = (
  options?: {
    repository?: EnvironmentRepository;
  } & EnvironmentGatewayDependencies,
): EnvironmentGateway => {
  const now = options?.now ?? (() => new Date().toISOString());
  const idFactory = options?.idFactory ?? ((prefix: string) => `${prefix}_${randomUUID().slice(0, 8)}`);
  const repository = options?.repository ?? createInMemoryRepository({ now, idFactory });

  return {
    create: (input: EnvironmentTwinCreateRequest): EnvironmentTwinRecord => repository.create(input),
    list: (): EnvironmentSummary[] => repository.list(),
    getBySlug: (slug: string): EnvironmentTwinRecord | null => repository.getBySlug(slug),
    upsertRecord: (slug: string, record: EnvironmentRecord): EnvironmentRecord | null => {
      const twin = repository.getBySlug(slug);
      if (!twin) return null;
      return repository.upsertRecord(twin.id, record);
    },
    addRelationship: (slug: string, input: EnvironmentRelationship): EnvironmentRelationship | null => {
      const twin = repository.getBySlug(slug);
      if (!twin) return null;
      return repository.addRelationship(twin.id, input);
    },
    addAccessPath: (slug: string, input: EnvironmentAccessPath): EnvironmentAccessPath | null => {
      const twin = repository.getBySlug(slug);
      if (!twin) return null;
      return repository.addAccessPath(twin.id, input);
    },
    addProcedure: (slug: string, input: EnvironmentProcedure): EnvironmentProcedure | null => {
      const twin = repository.getBySlug(slug);
      if (!twin) return null;
      return repository.addProcedure(twin.id, input);
    },
    getContextBundle: (slug: string, limit = 10): EnvironmentContextBundle | null => {
      const twin = repository.getBySlug(slug);
      if (!twin) return null;

      return EnvironmentContextBundleSchema.parse({
        environment: twin,
        records: repository.listRecords(twin.id, limit),
        relationships: repository.listRelationships(twin.id, limit),
        accessPaths: repository.listAccessPaths(twin.id, limit),
        procedures: repository.listProcedures(twin.id, limit),
      });
    },
  };
};

export type {
  EnvironmentAccessPath,
  EnvironmentContextBundle,
  EnvironmentMetadata,
  EnvironmentProcedure,
  EnvironmentRecord,
  EnvironmentRelationship,
  EnvironmentSummary,
  EnvironmentTwinCreateRequest,
  EnvironmentTwinRecord,
} from '@versa/shared';
