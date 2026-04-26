# Agent Task Card

- Issue URL: https://github.com/Deveadra/versa/issues/49
- Issue: #49
- Issue Title: [Redesign][WS4] Canonical memory hierarchy and gateway
- Parent Epic: #42
- Workstream: WS04

- Task Card ID: WS04-ISSUE49
- Task Card Name: memory-gateway
- Task Card File Name: ws04-issue-49-memory-gateway.md
- Task Card Path: docs/task-cards/active/ws04-issue-49-memory-gateway.md

- Status: Active
- Priority: High
- Agent Type: Roo

- Base Branch: main
- Branch: redesign/ws04-memory-gateway
- PR Title: redesign(ws04): implement canonical memory hierarchy and gateway

- Depends On: #43, #46, #47, #48
- Blocks: #50, #51, #52, #53, #54, #55, #56, #57

## Objective

Implement the canonical memory hierarchy and gateway so all durable memory access flows through one governed path.

## In Scope

- Create `packages/memory` as the canonical memory package
- Add typed shared contracts in `packages/shared` for:
  - session memory
  - episodic memory
  - semantic memory
  - procedural memory
  - memory metadata
  - memory gateway operations
- Add database structures and repository-layer support in `packages/database` as needed for the canonical memory hierarchy
- Add minimal integration points in `apps/core` and/or `apps/ai` only where needed to establish the governed memory path
- Add docs describing:
  - memory hierarchy
  - ownership and write path rules
  - read/write boundaries
  - provenance/confidence/retention expectations
- Keep implementation additive and bounded to memory hierarchy and gateway foundations only

## Out of Scope

- Full legacy Python memory migration
- Broad MCP exposure
- UI work
- Broad runtime rewrites unrelated to memory hierarchy/gateway foundations
- Self-improvement loop changes
- Unrelated cleanup

## Files/Areas to Inspect First

- `packages/memory/`
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

## Required Approach

1. Inspect current repo state before editing.
2. Read issue #49 first, then this task card.
3. Preserve working behavior unless the issue explicitly changes it.
4. Prefer additive contracts, package scaffolding, repository changes, and docs over broad refactors.
5. Keep the memory gateway canonical and singular.
6. Do not allow multiple uncontrolled memory write paths.
7. Keep the PR bounded to this issue only.

## Required Validation

- `pnpm install`
- `pnpm lint`
- `pnpm typecheck`
- `pnpm test`

## Deliverables

- memory-related code changes
- shared contracts for canonical memory hierarchy and gateway
- database/repository updates required for the memory foundation
- docs updates
- brief implementation notes in the PR body

## No-Touch Constraints

- Do not delete the legacy Python runtime
- Do not perform unrelated repo cleanup
- Do not rewrite adjacent subsystems unless strictly necessary for this issue
- Do not introduce placeholder memory code without documenting why it exists
- Do not drift into MCP gateway, approvals, frontend operator console, or broad self-improvement work
- Do not perform full legacy Python memory migration in this issue

## Acceptance Criteria

- `packages/memory` exists as the canonical memory package
- Shared memory contracts are stable and importable
- Memory hierarchy types are explicitly represented
- A governed memory gateway path is established
- Database/repository support is sufficient for the foundation being introduced
- Existing runtime behavior remains intact or improves without breakage

## Notes for Agent

This workstream depends on the architecture baseline from WS0, canonical config/dependency governance from WS1, telemetry foundation from WS2, and identity/doctrine groundwork from WS3. Read the GitHub issue first, then this task card, then inspect the repo. Extract `Base Branch` and `Branch` from this card. Create or switch to the work branch from the stated base branch only. Stay strictly within WS04 scope.
