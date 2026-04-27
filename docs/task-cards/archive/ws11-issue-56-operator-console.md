# Agent Task Card

- Issue URL: https://github.com/Deveadra/versa/issues/56
- Issue: #56
- Issue Title: [Redesign][WS11] Frontend operator console
- Parent Epic: #42
- Workstream: WS11

- Task Card ID: WS11-ISSUE56
- Task Card Name: operator-console
- Task Card File Name: ws11-issue-56-operator-console.md
- Task Card Path: docs/task-cards/active/ws11-issue-56-operator-console.md

- Status: Active
- Priority: High
- Agent Type: Roo

- Base Branch: main
- Branch: redesign/ws11-operator-console
- PR Title: redesign(ws11): build frontend operator console

- Depends On: #43, #46, #47, #48, #49, #50, #51, #52, #53, #54, #55
- Blocks: #57

## Objective

Build a useful operator-facing frontend surface for inspecting system state and runtime activity.

## In Scope

- Extend `apps/web` into the canonical operator-facing frontend surface
- Add or refine typed shared API contracts in `packages/shared` where needed for operator-console consumption
- Add frontend surfaces for foundational operator visibility into:
  - system health/status
  - workspaces
  - memory summaries
  - traces/logs
  - approvals
  - environment overview
- Consume backend data only through typed APIs and stable contracts
- Add docs describing:
  - operator console purpose
  - current information architecture
  - current screen/surface scope
  - what is intentionally deferred
- Add tests for foundational operator-console behavior where practical
- Keep implementation additive and bounded to the frontend operator-console foundations only

## Out of Scope

- Full product polish
- Bypassing typed APIs
- Unrelated backend refactors unless needed for UI consumption
- Broad runtime rewrites unrelated to operator-console foundations
- Full design-system overhaul
- Unrelated cleanup

## Files/Areas to Inspect First

- `apps/web/`
- `packages/shared/`
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
- `docs/task-cards/active/ws07-issue-52-environment-twin.md`
- `docs/task-cards/active/ws08-issue-53-approvals-trust-ladder.md`
- `docs/task-cards/active/ws09-issue-54-mcp-gateway.md`
- `docs/task-cards/active/ws10-issue-55-ai-convergence-bridge.md`

## Required Approach

1. Inspect current repo state before editing.
2. Read issue #56 first, then this task card.
3. Preserve working behavior unless the issue explicitly changes it.
4. Prefer additive UI surfaces, typed API consumption, docs, and tests over broad refactors.
5. Keep the operator console useful, inspectable, and grounded in real platform data.
6. Do not bypass typed APIs or stable contracts.
7. Keep the PR bounded to this issue only.

## Required Validation

- `pnpm install`
- `pnpm lint`
- `pnpm typecheck`
- `pnpm test`

## Deliverables

- operator-console-related code changes in `apps/web`
- any required shared API-contract updates
- tests for foundational operator-console behavior where practical
- docs updates
- brief implementation notes in the PR body

## No-Touch Constraints

- Do not delete the legacy Python runtime
- Do not perform unrelated repo cleanup
- Do not rewrite adjacent subsystems unless strictly necessary for this issue
- Do not introduce placeholder operator-console code without documenting why it exists
- Do not bypass typed APIs
- Do not drift into full product polish, broad backend rewrites, or unrelated design-system work

## Acceptance Criteria

- `apps/web` exposes a useful operator-facing frontend surface
- At least foundational system-state and runtime-activity views are present
- Shared API contracts remain typed and stable
- The frontend consumes backend data through typed APIs rather than ad hoc fetch patterns
- Existing runtime behavior remains intact or improves without breakage

## Notes for Agent

This workstream depends on the architecture baseline from WS0, canonical config/dependency governance from WS1, telemetry foundation from WS2, identity/doctrine groundwork from WS3, memory hierarchy/gateway groundwork from WS4, workspace-state groundwork from WS5, skills-engine groundwork from WS6, environment-twin groundwork from WS7, approvals/trust-ladder groundwork from WS8, MCP-gateway groundwork from WS9, and AI-service convergence/bridge groundwork from WS10. Read the GitHub issue first, then this task card, then inspect the repo. Extract `Base Branch` and `Branch` from this card. Create or switch to the work branch from the stated base branch only. Stay strictly within WS11 scope.
