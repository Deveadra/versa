# Agent Task Card

- Issue URL: https://github.com/Deveadra/versa/issues/52
- Issue: #52
- Issue Title: [Redesign][WS7] Environment twin and system map
- Parent Epic: #42
- Workstream: WS07

- Task Card ID: WS07-ISSUE52
- Task Card Name: environment-twin
- Task Card File Name: ws07-issue-52-environment-twin.md
- Task Card Path: docs/task-cards/active/ws07-issue-52-environment-twin.md

- Status: Active
- Priority: High
- Agent Type: Roo

- Base Branch: main
- Branch: redesign/ws07-environment-twin
- PR Title: redesign(ws07): add environment twin and system map

- Depends On: #43, #46, #47, #48, #49, #50, #51
- Blocks: #53, #54, #55, #56, #57

## Objective

Implement the environment twin so Ultron can reason over systems, services, access paths, and procedures.

## In Scope

- Create `packages/environment` as the canonical environment-twin package
- Add typed shared contracts in `packages/shared` for:
  - environment entities
  - systems
  - services
  - repos
  - dashboards
  - access paths
  - commands
  - procedures
  - environment relationships
  - environment metadata
- Add database structures and repository-layer support in `packages/database` as needed for durable environment-twin state
- Add minimal API or read-path integration only where needed to establish foundational access to environment-twin data
- Add docs describing:
  - environment twin purpose
  - environment twin ownership
  - entity types and relationships
  - how access paths and procedures are represented
  - how the environment twin differs from generic notes or memory
- Add tests for the foundational environment-twin contracts and persistence behavior
- Keep implementation additive and bounded to environment-twin foundations only

## Out of Scope

- Manual ingestion of every real environment entry
- UI-heavy work
- Collapsing environment into generic notes
- Broad runtime rewrites unrelated to environment-twin foundations
- MCP gateway implementation beyond any interfaces needed for future compatibility
- Unrelated cleanup

## Files/Areas to Inspect First

- `packages/environment/`
- `packages/shared/`
- `packages/database/`
- `apps/core/`
- `apps/ai/`
- `README.md`
- `docs/adr/`
- `docs/redesign/`
- `docs/task-cards/active/ws00-issue-43-architecture-baseline.md`
- `docs/task-cards/active/ws01-issue-46-config-governance.md`
- `docs/task-cards/active/ws02-issue-47-telemetry-foundation.md`
- `docs/task-cards/active/ws03-issue-48-identity-doctrine.md`
- `docs/task-cards/active/ws04-issue-49-memory-gateway.md`
- `docs/task-cards/active/ws05-issue-50-workspace-state.md`
- `docs/task-cards/active/ws06-issue-51-skills-engine.md`

## Required Approach

1. Inspect current repo state before editing.
2. Read issue #52 first, then this task card.
3. Preserve working behavior unless the issue explicitly changes it.
4. Prefer additive contracts, package scaffolding, repository changes, docs, and tests over broad refactors.
5. Keep environment-twin data structured, explicit, and queryable.
6. Do not blur environment-twin state with memory state or generic notes without clear contracts.
7. Keep the PR bounded to this issue only.

## Required Validation

- `pnpm install`
- `pnpm lint`
- `pnpm typecheck`
- `pnpm test`

## Deliverables

- environment-related code changes
- shared contracts for environment-twin foundations
- database/repository updates required for durable environment-twin state
- minimal API or read-path support where needed
- tests for foundational environment-twin behavior
- docs updates
- brief implementation notes in the PR body

## No-Touch Constraints

- Do not delete the legacy Python runtime
- Do not perform unrelated repo cleanup
- Do not rewrite adjacent subsystems unless strictly necessary for this issue
- Do not introduce placeholder environment code without documenting why it exists
- Do not drift into UI-heavy work, MCP gateway implementation, approvals, or frontend operator console work
- Do not collapse environment data into generic notes or unrelated memory structures

## Acceptance Criteria

- `packages/environment` exists as the canonical environment-twin package
- Shared environment contracts are stable and importable
- Environment entities, relationships, access paths, and procedures are explicitly represented
- Database/repository support is sufficient for the environment-twin foundation being introduced
- Minimal API or read-path support exists where needed for foundational access
- Existing runtime behavior remains intact or improves without breakage

## Notes for Agent

This workstream depends on the architecture baseline from WS0, canonical config/dependency governance from WS1, telemetry foundation from WS2, identity/doctrine groundwork from WS3, memory hierarchy/gateway groundwork from WS4, workspace-state groundwork from WS5, and skills-engine groundwork from WS6. Read the GitHub issue first, then this task card, then inspect the repo. Extract `Base Branch` and `Branch` from this card. Create or switch to the work branch from the stated base branch only. Stay strictly within WS07 scope.
