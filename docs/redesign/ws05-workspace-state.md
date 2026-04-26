# WS05 — Workspace state subsystem

Issue: `https://github.com/Deveadra/versa/issues/50`

## Canonical package decision

WS05 establishes one canonical workspace-state package: `@versa/workspaces` in [`packages/workspaces`](../../packages/workspaces).

- `@versa/workspaces` defines the governed gateway and repository contract for durable workspace state operations.
- Workspace state is explicitly separated from memory-state contracts to prevent semantic overlap and write-path drift.

## Workspace state contract surface

Workspace identity, state, metadata, checkpoint, summary, and context-bundle contracts are defined in [`packages/shared/src/index.ts`](../../packages/shared/src/index.ts):

- `WorkspaceIdentitySchema`
- `WorkspaceStateSchema`
- `WorkspaceMetadataSchema`
- `WorkspaceCheckpointSchema`
- `WorkspaceSummarySchema`
- `WorkspaceContextBundleSchema`

The state contract includes the required issue fields:

- current objective
- active blockers
- recent decisions
- important files
- known commands
- validated procedures
- next recommended actions

## Durable storage foundation

WS05 introduces durable SQLite storage for workspaces and checkpoints:

- table: `workspaces`
- table: `workspace_checkpoints`
- indexes for recency and checkpoint retrieval

Defined in [`packages/database/migrations/003_workspaces.sql`](../../packages/database/migrations/003_workspaces.sql), with repository support in [`packages/database/src/index.ts`](../../packages/database/src/index.ts) via `workspaceRepo()`.

## Minimal core API path

`@versa/core` now wires a minimal workspace API path using `createWorkspaceGateway({ repository: workspaceRepo(db) })` in [`apps/core/src/server.ts`](../../apps/core/src/server.ts).

Added endpoints:

- `GET /workspaces`
- `POST /workspaces`
- `GET /workspaces/:slug`
- `PATCH /workspaces/:slug/state`
- `POST /workspaces/:slug/activate`
- `POST /workspaces/:slug/checkpoints`
- `GET /workspaces/:slug/context`

This integration is additive and bounded to workspace-state foundations.

## Workspace lifecycle model

1. Create a named workspace (`slug`, `name`, optional repository/owner tags).
2. Update objective/state fields during execution loops.
3. Activate workspace when it becomes the current workstream.
4. Capture checkpoints for durable milestones and recovery.
5. Retrieve context bundles for continuity across sessions/tasks.

## WS05 boundaries

- No polished workspace UI.
- No automatic project discovery.
- No MCP gateway implementation.
- No unrelated memory refactors.
- Existing non-workspace runtime behavior remains intact.
