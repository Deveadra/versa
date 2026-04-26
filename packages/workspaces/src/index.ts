import { randomUUID } from 'node:crypto';
import {
  WorkspaceCheckpointCreateRequestSchema,
  WorkspaceCheckpointSchema,
  WorkspaceContextBundleSchema,
  WorkspaceCreateRequestSchema,
  WorkspaceRecordSchema,
  WorkspaceStatePatchSchema,
  WorkspaceSummarySchema,
  type WorkspaceCheckpoint,
  type WorkspaceCheckpointCreateRequest,
  type WorkspaceContextBundle,
  type WorkspaceCreateRequest,
  type WorkspaceRecord,
  type WorkspaceState,
  type WorkspaceStatePatch,
  type WorkspaceSummary,
} from '@versa/shared';

export type WorkspaceRepository = {
  create(input: WorkspaceCreateRequest): WorkspaceRecord;
  list(): WorkspaceSummary[];
  getBySlug(slug: string): WorkspaceRecord | null;
  updateState(workspaceId: string, patch: WorkspaceStatePatch): WorkspaceRecord | null;
  setActivated(workspaceId: string): WorkspaceRecord | null;
  createCheckpoint(
    workspaceId: string,
    input: WorkspaceCheckpointCreateRequest,
  ): WorkspaceCheckpoint | null;
  listCheckpoints(workspaceId: string, limit?: number): WorkspaceCheckpoint[];
};

export type WorkspaceGateway = {
  create(input: WorkspaceCreateRequest): WorkspaceRecord;
  list(): WorkspaceSummary[];
  getBySlug(slug: string): WorkspaceRecord | null;
  updateState(slug: string, patch: WorkspaceStatePatch): WorkspaceRecord | null;
  activate(slug: string): WorkspaceRecord | null;
  checkpoint(slug: string, input: WorkspaceCheckpointCreateRequest): WorkspaceCheckpoint | null;
  getContextBundle(slug: string, limit?: number): WorkspaceContextBundle | null;
};

type WorkspaceGatewayDependencies = {
  now?: () => string;
  idFactory?: (prefix: string) => string;
};

type InMemoryWorkspaceStore = {
  byId: Map<string, WorkspaceRecord>;
  bySlug: Map<string, string>;
  checkpoints: Map<string, WorkspaceCheckpoint[]>;
};

const toSummary = (workspace: WorkspaceRecord): WorkspaceSummary =>
  WorkspaceSummarySchema.parse({
    id: workspace.id,
    slug: workspace.slug,
    name: workspace.name,
    currentObjective: workspace.state.currentObjective,
    activeBlockerCount: workspace.state.activeBlockers.filter(
      (b: WorkspaceState['activeBlockers'][number]) => b.status === 'active',
    ).length,
    nextActionCount: workspace.state.nextRecommendedActions.length,
    updatedAt: workspace.metadata.updatedAt,
    lastActivatedAt: workspace.metadata.lastActivatedAt,
  });

const patchState = (
  current: WorkspaceState,
  patch: WorkspaceStatePatch,
  timestamp: string,
): WorkspaceState => ({
  ...current,
  ...patch,
  updatedAt: timestamp,
});

const createInMemoryStore = (): InMemoryWorkspaceStore => ({
  byId: new Map(),
  bySlug: new Map(),
  checkpoints: new Map(),
});

const createInMemoryRepository = (
  deps: Required<WorkspaceGatewayDependencies>,
  store = createInMemoryStore(),
): WorkspaceRepository => ({
  create: (input: WorkspaceCreateRequest): WorkspaceRecord => {
    const parsed = WorkspaceCreateRequestSchema.parse(input);
    if (store.bySlug.has(parsed.slug)) {
      throw new Error(`workspace slug already exists: ${parsed.slug}`);
    }

    const timestamp = deps.now();
    const workspace = WorkspaceRecordSchema.parse({
      id: deps.idFactory('wrk'),
      slug: parsed.slug,
      name: parsed.name,
      repository: parsed.repository,
      metadata: {
        owner: parsed.metadata.owner,
        tags: parsed.metadata.tags ?? [],
        source: parsed.metadata.source ?? 'manual',
        createdAt: timestamp,
        updatedAt: timestamp,
        lastActivatedAt: parsed.metadata.lastActivatedAt,
      },
      state: {
        ...parsed.state,
        updatedAt: timestamp,
      },
    });

    store.byId.set(workspace.id, workspace);
    store.bySlug.set(workspace.slug, workspace.id);
    store.checkpoints.set(workspace.id, []);

    return workspace;
  },

  list: (): WorkspaceSummary[] =>
    Array.from(store.byId.values())
      .map(toSummary)
      .sort((a, b) => b.updatedAt.localeCompare(a.updatedAt)),

  getBySlug: (slug: string): WorkspaceRecord | null => {
    const workspaceId = store.bySlug.get(slug);
    if (!workspaceId) return null;
    return store.byId.get(workspaceId) ?? null;
  },

  updateState: (workspaceId: string, patch: WorkspaceStatePatch): WorkspaceRecord | null => {
    const current = store.byId.get(workspaceId);
    if (!current) return null;

    const parsedPatch = WorkspaceStatePatchSchema.parse(patch);
    const timestamp = deps.now();
    const updated = WorkspaceRecordSchema.parse({
      ...current,
      metadata: {
        ...current.metadata,
        updatedAt: timestamp,
      },
      state: patchState(current.state, parsedPatch, timestamp),
    });

    store.byId.set(workspaceId, updated);
    return updated;
  },

  setActivated: (workspaceId: string): WorkspaceRecord | null => {
    const current = store.byId.get(workspaceId);
    if (!current) return null;
    const timestamp = deps.now();
    const updated = WorkspaceRecordSchema.parse({
      ...current,
      metadata: {
        ...current.metadata,
        updatedAt: timestamp,
        lastActivatedAt: timestamp,
      },
    });
    store.byId.set(workspaceId, updated);
    return updated;
  },

  createCheckpoint: (
    workspaceId: string,
    input: WorkspaceCheckpointCreateRequest,
  ): WorkspaceCheckpoint | null => {
    const current = store.byId.get(workspaceId);
    if (!current) return null;

    const parsed = WorkspaceCheckpointCreateRequestSchema.parse(input);
    const snapshot = parsed.snapshot
      ? {
          ...parsed.snapshot,
          updatedAt: deps.now(),
        }
      : current.state;

    const checkpoint = WorkspaceCheckpointSchema.parse({
      id: deps.idFactory('wcp'),
      workspaceId,
      summary: parsed.summary,
      snapshot,
      createdAt: deps.now(),
      createdBy: parsed.createdBy,
    });

    const list = store.checkpoints.get(workspaceId) ?? [];
    list.unshift(checkpoint);
    store.checkpoints.set(workspaceId, list);
    return checkpoint;
  },

  listCheckpoints: (workspaceId: string, limit = 10): WorkspaceCheckpoint[] => {
    const list = store.checkpoints.get(workspaceId) ?? [];
    return list.slice(0, Math.max(1, limit));
  },
});

export const createWorkspaceGateway = (
  options?: {
    repository?: WorkspaceRepository;
  } & WorkspaceGatewayDependencies,
): WorkspaceGateway => {
  const now = options?.now ?? (() => new Date().toISOString());
  const idFactory = options?.idFactory ?? ((prefix: string) => `${prefix}_${randomUUID().slice(0, 8)}`);

  const repository = options?.repository ?? createInMemoryRepository({ now, idFactory });

  return {
    create: (input: WorkspaceCreateRequest): WorkspaceRecord => repository.create(input),
    list: (): WorkspaceSummary[] => repository.list(),
    getBySlug: (slug: string): WorkspaceRecord | null => repository.getBySlug(slug),
    updateState: (slug: string, patch: WorkspaceStatePatch): WorkspaceRecord | null => {
      const workspace = repository.getBySlug(slug);
      if (!workspace) return null;
      return repository.updateState(workspace.id, patch);
    },
    activate: (slug: string): WorkspaceRecord | null => {
      const workspace = repository.getBySlug(slug);
      if (!workspace) return null;
      return repository.setActivated(workspace.id);
    },
    checkpoint: (slug: string, input: WorkspaceCheckpointCreateRequest): WorkspaceCheckpoint | null => {
      const workspace = repository.getBySlug(slug);
      if (!workspace) return null;
      return repository.createCheckpoint(workspace.id, input);
    },
    getContextBundle: (slug: string, limit = 5): WorkspaceContextBundle | null => {
      const workspace = repository.getBySlug(slug);
      if (!workspace) return null;

      const summary = toSummary(workspace);
      const recentCheckpoints = repository.listCheckpoints(workspace.id, limit);

      return WorkspaceContextBundleSchema.parse({
        workspace,
        summary,
        recentCheckpoints,
      });
    },
  };
};

export type {
  WorkspaceBlocker,
  WorkspaceCheckpoint,
  WorkspaceCheckpointCreateRequest,
  WorkspaceContextBundle,
  WorkspaceCreateRequest,
  WorkspaceRecord,
  WorkspaceState,
  WorkspaceStatePatch,
  WorkspaceSummary,
} from '@versa/shared';
