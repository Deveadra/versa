# Agent Task Card

- Issue URL: https://github.com/Deveadra/versa/issues/47
- Issue: #47
- Issue Title: [Redesign][WS2] Telemetry and observability foundation
- Parent Epic: #42
- Workstream: WS02

- Task Card ID: WS02-ISSUE47
- Task Card Name: telemetry-foundation
- Task Card File Name: ws02-issue-47-telemetry-foundation.md
- Task Card Path: docs/task-cards/active/ws02-issue-47-telemetry-foundation.md

- Status: Active
- Priority: High
- Agent Type: Roo

- Base Branch: main
- Branch: redesign/ws02-telemetry-foundation
- PR Title: redesign(ws02): add telemetry and observability foundation

- Depends On: #43, #46
- Blocks: #53, #54, #55, #56, #57

## Objective

Establish the telemetry, logging, and execution-trace foundation required to make all later redesign work observable, debuggable, and auditable.

## In Scope

- Expand `packages/logging` into a structured logging foundation
- Add shared telemetry and trace contracts under `packages/shared`
- Define conventions for:
  - trace IDs
  - correlation IDs
  - run IDs
  - actor/source metadata
  - event types
- Add baseline telemetry wiring to:
  - `apps/core/src/server.ts`
  - `apps/ai/src/server.ts`
- Add docs describing the telemetry model, traceability expectations, and current observability boundaries
- Keep implementation additive and bounded to foundational telemetry work only

## Out of Scope

- Full operator UI or dashboards
- Vendor-specific observability platform rollout
- Broad approvals or MCP implementation beyond interfaces/contracts needed for telemetry compatibility
- Runtime rewrites unrelated to telemetry/logging foundations
- Unrelated cleanup

## Files/Areas to Inspect First

- `packages/logging/`
- `packages/shared/`
- `apps/core/src/server.ts`
- `apps/ai/src/server.ts`
- `README.md`
- `docs/adr/`
- `docs/redesign/`
- `docs/task-cards/active/ws00-issue-43-architecture-baseline.md`
- `docs/task-cards/active/ws01-issue-46-config-governance.md`

## Required Approach

1. Inspect current repo state before editing.
2. Read issue #47 first, then this task card.
3. Preserve working behavior unless the issue explicitly changes it.
4. Prefer additive logging/contracts/instrumentation over broad refactors.
5. Keep telemetry contracts typed and importable.
6. Add or update docs for the workstream.
7. Keep the PR bounded to this issue only.

## Required Validation

- `pnpm install`
- `pnpm lint`
- `pnpm typecheck`
- `pnpm test`

## Deliverables

- telemetry-related code changes
- shared contracts for tracing/logging foundation
- docs updates
- brief implementation notes in the PR body

## No-Touch Constraints

- Do not delete the legacy Python runtime
- Do not perform unrelated repo cleanup
- Do not rewrite adjacent subsystems unless strictly necessary for this issue
- Do not introduce placeholder telemetry code without documenting why it exists
- Do not drift into approvals, memory hierarchy, skills, MCP gateway, or frontend operator console work

## Acceptance Criteria

- `apps/core` and `apps/ai` emit consistent structured logs
- Shared telemetry contracts are stable and importable
- Trace and event metadata conventions are documented clearly
- Logging/telemetry changes remain additive and bounded
- Existing runtime behavior remains intact or improves without breakage

## Notes for Agent

This workstream depends on the architecture baseline from WS0 and the canonical config/dependency governance from WS1. Read the GitHub issue first, then this task card, then inspect the repo. Extract `Base Branch` and `Branch` from this card. Create or switch to the work branch from the stated base branch only. Stay strictly within WS02 scope.
