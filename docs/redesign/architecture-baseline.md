# Redesign architecture baseline (WS0 / issue #43)

This document is the execution baseline for redesign workstreams that depend on issue #43.

## Scope and intent

- establish one contract-first architecture baseline
- document current-state vs target-state ownership
- define naming standards and dependency directions
- define phased migration rules so TypeScript and legacy Python can coexist safely

## Current-state vs target-state map

| Area | Current state | Target state |
| --- | --- | --- |
| `apps/core` | Express API with domain routes and DB repositories | Core orchestration/API boundary for local-first runtime |
| `apps/ai` | Adapter service with health + placeholder capabilities | AI capability adapter behind explicit contracts and policy |
| `apps/web` | Next.js shell consuming core API | Operator console and user-facing web surface |
| `packages/shared` | Canonical enums + schemas + event contract | Canonical cross-workstream contract authority |
| `packages/config` | Minimal runtime config parser | Canonical configuration + precedence governance |
| `packages/database` | SQLite repos/migrations/seed for monorepo stack | Durable persistence boundary and migration authority |
| `src/**` (legacy Python runtime) | Existing production/legacy runtime | Preserved during redesign; bridge/migration happens in later WS |

## Package and app ownership

- `apps/core`: owns API orchestration and non-UI runtime endpoints
- `apps/ai`: owns AI adapter invocation boundary only
- `apps/web`: owns UI/operator interactions and presentation logic
- `packages/shared`: owns contract schemas and shared type surfaces
- `packages/config`: owns config schema, defaults, environment naming/precedence
- `packages/database`: owns SQL migrations, schema persistence, repository data access
- `packages/logging`: owns structured telemetry logging primitives
- `packages/security`: owns policy/security primitives and redaction boundaries
- `packages/integrations`: owns external connector abstractions

## Stable platform contracts (defined in `@versa/shared`)

The following contracts are established for downstream workstreams:

- identity/doctrine: `IdentityProfileSchema`, `DoctrinePolicySchema`
- memory gateway: `MemoryQuerySchema`, `MemoryRecordSchema`
- workspace state: `WorkspaceStateSchema`
- skills: `SkillDescriptorSchema`
- environment twin: `EnvironmentTwinSchema`
- approvals/trust ladder: `ApprovalRequestSchema`, `ApprovalDecisionSchema`, `TrustLevelEnum`
- telemetry events: `TelemetryEventSchema`, `TelemetryEventNameEnum`
- MCP gateway registry: `McpCapabilitySchema`, `McpServerRegistrationSchema`

## Naming standards

### Packages

- workspace package names use `@versa/<domain>`
- package directory names use kebab-case (`packages/shared`, `packages/database`)
- contract modules use kebab-case filenames (`identity-doctrine.ts`)

### API paths

- REST endpoints use lowercase, kebab-case path segments
- domain grouping uses top-level nouns (`/tasks`, `/goals`, `/study`, `/jobs`)

### Event types

- dot-delimited lower-case format: `<domain>.<entity>.<action>`
- examples: `task.created`, `approval.requested`, `mcp.capability.invoked`

### Storage tables

- snake_case plural table names for domain entities (`tasks`, `study_assignments`)
- snake_case singular `_id` foreign keys (`lead_id`, `linked_goal_id`)

## Dependency graph and import rules

Allowed directions:

1. `apps/*` may import from `packages/*`
2. `packages/database` may import from `packages/shared`
3. `packages/config`, `packages/logging`, `packages/security`, `packages/integrations` may import from `packages/shared`

Forbidden directions:

- `packages/*` must not import from any `apps/*`
- `packages/shared` must not import from app-local or infra-specific packages
- `apps/web` must not import directly from `packages/database`
- redesign TypeScript packages/apps must not import legacy Python runtime modules directly

## Phased migration rules

1. **Contract-first:** add/adjust shared contracts before introducing deep runtime coupling.
2. **Additive changes:** prefer new package/module surfaces over destructive rewrites.
3. **Legacy preservation:** do not delete or heavily rewrite legacy Python runtime in WS0.
4. **Boundary integrity:** treat `apps/ai` as optional adapter; `apps/core` must remain functional without AI.
5. **Workstream discipline:** downstream WS cards implement against these contracts and must not redefine cross-cutting schemas ad hoc.

## References

- ADR baseline decision: `docs/adrs/ADR-009.md`
- Issue authority: https://github.com/Deveadra/versa/issues/43
- Task card: `docs/task-cards/active/ws00-issue-43-architecture-baseline.md`

