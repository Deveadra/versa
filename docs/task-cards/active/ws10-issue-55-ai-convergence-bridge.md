# Agent Task Card

- Issue URL: https://github.com/Deveadra/versa/issues/55
- Issue: #55
- Issue Title: [Redesign][WS10] AI service convergence and legacy Python bridge
- Parent Epic: #42
- Workstream: WS10

- Task Card ID: WS10-ISSUE55
- Task Card Name: ai-convergence-bridge
- Task Card File Name: ws10-issue-55-ai-convergence-bridge.md
- Task Card Path: docs/task-cards/active/ws10-issue-55-ai-convergence-bridge.md

- Status: Active
- Priority: High
- Agent Type: Roo

- Base Branch: main
- Branch: redesign/ws10-ai-convergence-bridge
- PR Title: redesign(ws10): converge AI service and add legacy Python bridge

- Depends On: #43, #46, #47, #48, #49, #50, #51, #52, #53, #54
- Blocks: #56, #57

## Objective

Converge the AI-facing service layer and add a controlled bridge to the legacy Python runtime.

## In Scope

- Refine `apps/ai` into the canonical AI-facing service entry point for the new platform direction
- Add typed shared contracts in `packages/shared` for:
  - bridge request and response shapes
  - bridge capability declarations
  - bridge health/status
  - AI service boundary contracts
  - compatibility or adapter result structures
- Add or refine config support in `packages/config` as needed for bridge and convergence behavior
- Create `packages/bridge` only if it is necessary to keep the bridge boundary explicit and clean
- Add docs describing:
  - what remains owned by the legacy Python runtime
  - what becomes owned by the converged AI-facing TypeScript service layer
  - what the bridge is allowed to do
  - what the bridge must not do
  - how the bridge fits into the phased migration strategy
- Add tests for foundational bridge and AI-service convergence behavior
- Keep implementation additive and bounded to AI convergence and legacy bridge foundations only

## Out of Scope

- Replacing the Python runtime wholesale
- Deleting legacy code
- Collapsing all future AI behavior into one oversized service
- Broad runtime rewrites unrelated to AI-service convergence and bridge foundations
- Frontend/operator-console work
- Unrelated cleanup

## Files/Areas to Inspect First

- `apps/ai/`
- `packages/shared/`
- `packages/config/`
- `packages/bridge/`
- `README.md`
- `docs/adr/`
- `docs/redesign/`
- `src/base/`
- `run.py`
- `docs/task-cards/active/ws00-issue-43-architecture-baseline.md`
- `docs/task-cards/active/ws01-issue-46-config-governance.md`
- `docs/task-cards/active/ws02-issue-47-telemetry-foundation.md`
- `docs/task-cards/active/ws03-issue-48-identity-doctrine.md`
- `docs/task-cards/active/ws04-issue-49-memory-gateway.md`
- `docs/task-cards/active/ws05-issue-50-workspace-state.md`
- `docs/task-cards/active/ws06-issue-51-skills-engine.md`
- `docs/task-cards/active/ws07-issue-52-environment-twin.md`
- `docs/task-cards/active/ws08-issue-53-approvals-trust-ladder.md`
- `docs/task-cards/active/ws09-issue-54-mcp-gateway.md`

## Required Approach

1. Inspect current repo state before editing.
2. Read issue #55 first, then this task card.
3. Preserve working behavior unless the issue explicitly changes it.
4. Prefer additive contracts, adapters, service-boundary changes, docs, and tests over broad refactors.
5. Keep the AI-facing TypeScript service boundary explicit and typed.
6. Keep the legacy Python bridge narrow, intentional, and reversible.
7. Do not let this issue become a stealth rewrite of the legacy runtime.
8. Keep the PR bounded to this issue only.

## Required Validation

- `pnpm install`
- `pnpm lint`
- `pnpm typecheck`
- `pnpm test`
- any bridge smoke tests created by the implementation

## Deliverables

- AI-service and bridge-related code changes
- shared contracts for foundational bridge and convergence behavior
- config updates required for bridge/convergence foundations
- tests for foundational bridge behavior
- docs updates
- brief implementation notes in the PR body

## No-Touch Constraints

- Do not replace the Python runtime wholesale
- Do not delete legacy code
- Do not collapse all future AI behavior into one oversized service
- Do not perform unrelated repo cleanup
- Do not rewrite adjacent subsystems unless strictly necessary for this issue
- Do not introduce placeholder bridge code without documenting why it exists
- Do not drift into frontend/operator-console work or unrelated platform rewrites in this issue

## Acceptance Criteria

- `apps/ai` has a clearer converged responsibility as the AI-facing service layer
- Shared bridge and AI-service contracts are stable and importable
- Config support exists where needed for foundational bridge behavior
- The boundary between legacy Python ownership and new TypeScript service ownership is documented clearly
- Foundational bridge tests or smoke tests exist where needed
- Existing runtime behavior remains intact or improves without breakage

## Notes for Agent

This workstream depends on the architecture baseline from WS0, canonical config/dependency governance from WS1, telemetry foundation from WS2, identity/doctrine groundwork from WS3, memory hierarchy/gateway groundwork from WS4, workspace-state groundwork from WS5, skills-engine groundwork from WS6, environment-twin groundwork from WS7, approvals/trust-ladder groundwork from WS8, and MCP-gateway groundwork from WS9. Read the GitHub issue first, then this task card, then inspect the repo. Extract `Base Branch` and `Branch` from this card. Create or switch to the work branch from the stated base branch only. Stay strictly within WS10 scope.
