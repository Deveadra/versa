# WS03 — Identity and doctrine subsystem

Issue: `https://github.com/Deveadra/versa/issues/48`

## Canonical package decision

WS03 establishes one canonical identity/doctrine package: `@versa/doctrine` in [`packages/doctrine`](../../packages/doctrine).

- `packages/doctrine` is the single package for doctrine loading, validation, and retrieval.
- `packages/identity` is intentionally not created to avoid split ownership and contract drift.

## Doctrine contract surface

Canonical doctrine schemas and types are defined in `@versa/shared` at [`packages/shared/src/index.ts`](../../packages/shared/src/index.ts):

- `DoctrineSchema`
- `DoctrineResponseStyleSchema`
- `DoctrineEscalationRuleSchema`
- `DoctrineAutonomyBoundarySchema`
- `DoctrineSafetyRuleSchema`
- `DoctrineDecisionPriorityEnum`
- `DoctrineEscalationSeverityEnum`

These contracts are designed to be imported by runtime packages and apps as stable doctrine interfaces.

## Doctrine package responsibilities

`@versa/doctrine` exposes a minimal, additive runtime read path:

- `parseDoctrine(input)`
  - validates unknown doctrine payloads against `DoctrineSchema`
- `loadDoctrine({ filePath, enabled, fallback })`
  - loads doctrine from file when enabled/path is provided
  - falls back to canonical in-package default doctrine when disabled/path missing
- `loadDoctrineFromFile(path)`
  - deterministic file loader for doctrine JSON
- `createDoctrineStore(seed)` / `updateDoctrine(store, next)` / `getDoctrine(store)`
  - in-memory doctrine current+history strategy for version/change tracking expectations

Default doctrine is owned in [`packages/doctrine/src/default-doctrine.ts`](../../packages/doctrine/src/default-doctrine.ts).

## Ownership and change strategy

Doctrine ownership metadata is embedded in the doctrine contract (`ownership.team`, `ownership.maintainers`) and each doctrine snapshot includes:

- `version`
- `metadata.createdAt`
- `metadata.updatedAt`
- `metadata.changeSummary`

This gives downstream workstreams a stable baseline for doctrine versioning and change history without requiring a full persistence subsystem in WS03.

## Config touchpoints (minimal)

WS03 adds minimal doctrine config keys in `@versa/config`:

- `FEATURE_DOCTRINE_ENABLED` (default: `true`)
- `DOCTRINE_PATH` (default: `state/doctrine.json`)

These keys provide bounded feature-gate/path configuration for runtime doctrine loading without forcing immediate runtime rewrites.

## Boundaries for WS03

- No prompt-engineering overhaul.
- No legacy Python personality rewrite.
- No implementation drift into memory/skills/approvals/MCP/environment/operator-console workstreams.
- Existing runtime behavior in `apps/core` and `apps/ai` remains unchanged in this slice.
