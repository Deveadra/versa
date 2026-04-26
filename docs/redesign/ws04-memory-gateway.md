# WS04 — Canonical memory hierarchy and gateway

Issue: `https://github.com/Deveadra/versa/issues/49`

## Canonical package decision

WS04 establishes one canonical memory package: `@versa/memory` in [`packages/memory`](../../packages/memory).

- `@versa/memory` defines the single gateway contract for durable memory read/write/consolidation.
- Runtime paths that persist durable memories must flow through this gateway + repository pairing.

## Memory hierarchy model

The hierarchy is explicitly modeled in shared contracts via `MemoryTierEnum` in [`packages/shared/src/index.ts`](../../packages/shared/src/index.ts):

- `session`
- `episodic`
- `semantic`
- `procedural`

This keeps tier semantics explicit and importable across packages/apps.

## Canonical gateway operations

`@versa/memory` exposes `createMemoryGateway()` in [`packages/memory/src/index.ts`](../../packages/memory/src/index.ts) with four governed operations:

- `write(input)`
- `read(query)`
- `getById(memoryId)`
- `consolidate(input)`

When constructed with a repository adapter (`repository`), all durable writes are routed through the repository implementation, preventing uncontrolled write paths.

## Metadata model

`MemoryMetadataSchema` in [`packages/shared/src/index.ts`](../../packages/shared/src/index.ts) includes:

- `confidence` (0..1)
- `source` (`manual|imported|system|ai`)
- `sensitivity`
- `retention` (`strategy`, optional `ttlDays`, optional `decayRate`)
- `provenance` (`actor`, optional trace/event/subsystem/source lineage)
- `tags`

These fields form the required governance envelope for durable memory attribution and classification.

## Durable storage foundation

WS04 introduces durable memory storage in SQLite:

- table: `memories`
- columns: id, tier, summary, content JSON, metadata JSON, created/updated/accessed timestamps
- indexes: tier+updated, updated desc

Defined in [`packages/database/migrations/001_init.sql`](../../packages/database/migrations/001_init.sql), with repository support in [`packages/database/src/index.ts`](../../packages/database/src/index.ts) via `memoryRepo()`.

## Minimal integration path

`@versa/core` now wires one governed path:

- construct repository: `memoryRepo(db)`
- construct gateway: `createMemoryGateway({ repository: memoryRepo(db) })`
- expose minimal endpoints:
  - `GET /memory`
  - `POST /memory`
  - `POST /memory/consolidate`

Implementation is in [`apps/core/src/server.ts`](../../apps/core/src/server.ts). This is additive and does not rewrite existing runtime surfaces.

## Consolidation behavior (foundation)

Consolidation promotes repeated/source-linked memories into durable tiers (`semantic` or `procedural`) while preserving lineage (`sourceMemoryIds`) in provenance metadata.

This issue introduces the foundation contract and path only; advanced heuristics/policies remain future workstreams.

## WS04 boundaries

- No full legacy Python memory migration.
- No broad MCP memory endpoint surface.
- No UI or unrelated subsystem rewrites.
- Existing non-memory runtime behavior remains intact.
