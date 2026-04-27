# WS07 — Environment twin and system map

Issue: `https://github.com/Deveadra/versa/issues/52`

## Canonical package decision

WS07 establishes one canonical environment-twin package: `@versa/environment` in [`packages/environment`](../../packages/environment).

- `@versa/environment` defines the governed gateway and repository contract for durable environment-twin operations.
- Environment twin state is explicitly separated from memory-state contracts to preserve structured operational data and explicit query paths.

## Environment contract surface

Environment twin contracts are defined in [`packages/shared/src/index.ts`](../../packages/shared/src/index.ts):

- `EnvironmentRecordSchema`
- `EnvironmentRelationshipSchema`
- `EnvironmentAccessPathSchema`
- `EnvironmentProcedureSchema`
- `EnvironmentTwinRecordSchema`
- `EnvironmentSummarySchema`
- `EnvironmentContextBundleSchema`

These contracts provide explicit structures for:

- machines, services, dashboards, repositories, access paths, commands, procedures, and environment entities
- relationship edges between entities
- access paths and prerequisites
- procedural steps and expected outcomes

## Durable storage foundation

WS07 introduces durable SQLite storage for environment twins and linked operational entities:

- table: `environments`
- table: `environment_records`
- table: `environment_relationships`
- table: `environment_access_paths`
- table: `environment_procedures`

Defined in [`packages/database/migrations/004_environment_twin.sql`](../../packages/database/migrations/004_environment_twin.sql), with repository support in [`packages/database/src/index.ts`](../../packages/database/src/index.ts) via `environmentRepo()`.

## Minimal core API path

`@versa/core` now wires a minimal environment-twin API path using `createEnvironmentGateway({ repository: environmentRepo(db) })` in [`apps/core/src/server.ts`](../../apps/core/src/server.ts).

Added endpoints:

- `GET /environments`
- `POST /environments`
- `GET /environments/:slug`
- `PUT /environments/:slug/records`
- `POST /environments/:slug/relationships`
- `POST /environments/:slug/access-paths`
- `POST /environments/:slug/procedures`
- `GET /environments/:slug/context`

This integration is additive and bounded to environment-twin foundations.

## End-to-end example shape

One realistic environment can be represented end to end by:

1. Creating an environment twin for a target context (for example `prod/us-east-1`).
2. Upserting service and machine records.
3. Encoding dependency/access relationships between those records.
4. Attaching access path definitions for operational entry points.
5. Attaching a validated procedure with ordered steps for standard operations.
6. Reading a context bundle for durable operator-facing retrieval.

## WS07 boundaries

- No UI-heavy implementation.
- No manual ingestion of every environment entry.
- No MCP gateway implementation work beyond compatible contracts.
- No collapse into generic notes or memory structures.
- No unrelated runtime rewrites or cleanup.
