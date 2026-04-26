# Agent Task Card

- Issue URL: https://github.com/Deveadra/versa/issues/50
- Issue: #50
- Issue Title: [Redesign][WS5] Workspace state subsystem
- Parent Epic: #42
- Workstream: WS05

- Task Card ID: WS05-ISSUE50
- Task Card Name: workspace-state
- Task Card File Name: ws05-issue-50-workspace-state.md
- Task Card Path: docs/task-cards/active/ws05-issue-50-workspace-state.md

- Status: Active
- Priority: High
- Agent Type: Roo

- Base Branch: main
- Branch: redesign/ws05-workspace-state
- PR Title: redesign(ws05): add workspace state subsystem

- Depends On: #43, #46, #47, #48, #49
- Blocks: #51, #52, #53, #54, #55, #56, #57

## Objective

Implement durable named workspace state for project continuity.

## In Scope

- Create `packages/workspaces` as the canonical workspace-state package
- Add typed shared contracts in `packages/shared` for:
  - workspace identity
  - workspace state
  - workspace metadata
  - workspace checkpoints
  - workspace summaries
  - workspace context bundles
- Add database structures and repository-layer support in `packages/database` as needed for durable workspace state
- Add minimal `apps/core` APIs required to read, create, update, and retrieve workspace state
- Add docs describing:
  - workspace purpose
  - workspace ownership
  - workspace lifecycle
  - required fields
  - how workspace state supports continuity across sessions and tasks
- Keep implementation additive and bounded to workspace-state foundations only

## Out of Scope

- Polished workspace UI
- Automatic project discovery
- Unrelated memory refactors
- Broad runtime rewrites unrelated to workspace-state foundations
- MCP exposure beyond any interfaces needed for future compatibility
- Unrelated cleanup

## Files/Areas to Inspect First

- `packages/workspaces/`
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

## Required Approach

1. Inspect current repo state before editing.
2. Read issue #50 first, then this task card.
3. Preserve working behavior unless the issue explicitly changes it.
4. Prefer additive contracts, package scaffolding, repository changes, and docs over broad refactors.
5. Keep workspace state durable, explicit, and queryable.
6. Do not blur workspace state and memory state without clear contracts.
7. Keep the PR bounded to this issue only.

## Required Validation

- `pnpm install`
- `pnpm lint`
- `pnpm typecheck`
- `pnpm test`

## Deliverables

- workspace-related code changes
- shared contracts for workspace-state foundations
- database/repository updates required for durable workspace state
- minimal `apps/core` API support for workspace state
- docs updates
- brief implementation notes in the PR body

## No-Touch Constraints

- Do not delete the legacy Python runtime
- Do not perform unrelated repo cleanup
- Do not rewrite adjacent subsystems unless strictly necessary for this issue
- Do not introduce placeholder workspace code without documenting why it exists
- Do not drift into polished UI, automatic project discovery, MCP gateway implementation, or approvals work
- Do not perform unrelated memory refactors in this issue

## Acceptance Criteria

- `packages/workspaces` exists as the canonical workspace-state package
- Shared workspace contracts are stable and importable
- Durable workspace state is explicitly represented and queryable
- Database/repository support is sufficient for the workspace-state foundation being introduced
- Minimal `apps/core` APIs exist where needed for workspace operations
- Existing runtime behavior remains intact or improves without breakage

## Notes for Agent

This workstream depends on the architecture baseline from WS0, canonical config/dependency governance from WS1, telemetry foundation from WS2, identity/doctrine groundwork from WS3, and memory hierarchy/gateway groundwork from WS4. Read the GitHub issue first, then this task card, then inspect the repo. Extract `Base Branch` and `Branch` from this card. Create or switch to the work branch from the stated base branch only. Stay strictly within WS05 scope.
