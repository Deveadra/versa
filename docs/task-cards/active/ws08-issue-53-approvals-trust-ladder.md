# Agent Task Card

- Issue URL: https://github.com/Deveadra/versa/issues/53
- Issue: #53
- Issue Title: [Redesign][WS8] Approvals, trust ladder, and action policy
- Parent Epic: #42
- Workstream: WS08

- Task Card ID: WS08-ISSUE53
- Task Card Name: approvals-trust-ladder
- Task Card File Name: ws08-issue-53-approvals-trust-ladder.md
- Task Card Path: docs/task-cards/active/ws08-issue-53-approvals-trust-ladder.md

- Status: Active
- Priority: High
- Agent Type: Roo

- Base Branch: main
- Branch: redesign/ws08-approvals-trust-ladder
- PR Title: redesign(ws08): add approvals, trust ladder, and action policy

- Depends On: #43, #46, #47, #48, #49, #50, #51, #52
- Blocks: #54, #55, #56, #57

## Objective

Implement trust levels, approval policy, and enforcement hooks for safe bounded action.

## In Scope

- Create `packages/approvals` as the canonical approvals and action-policy package
- Add typed shared contracts in `packages/shared` for:
  - trust levels
  - approval requests
  - approval decisions
  - action policy rules
  - action classifications
  - approval results
  - audit metadata
  - enforcement outcomes
- Add enforcement hooks in `apps/core` and/or `apps/ai` only where needed to establish the foundational approval and policy path
- Add telemetry linkage for approval and policy events where required for traceability
- Add docs describing:
  - trust ladder purpose
  - approval policy purpose
  - approval flow
  - enforcement boundaries
  - how approval logic differs from prompt-only behavior
- Add tests for foundational approval-policy contracts and enforcement behavior
- Keep implementation additive and bounded to approvals/trust-ladder foundations only

## Out of Scope

- Full approval UI
- Enabling high-autonomy modes by default
- Implicit prompt-only policy logic
- Broad runtime rewrites unrelated to approvals/trust-ladder foundations
- Frontend/operator-console work
- Unrelated cleanup

## Files/Areas to Inspect First

- `packages/approvals/`
- `packages/shared/`
- `packages/logging/`
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

## Required Approach

1. Inspect current repo state before editing.
2. Read issue #53 first, then this task card.
3. Preserve working behavior unless the issue explicitly changes it.
4. Prefer additive contracts, package scaffolding, enforcement hooks, telemetry linkage, docs, and tests over broad refactors.
5. Keep approvals, trust levels, and policy enforcement explicit and typed.
6. Do not let approval behavior remain implicit, vague, or prompt-only.
7. Keep the PR bounded to this issue only.

## Required Validation

- `pnpm install`
- `pnpm lint`
- `pnpm typecheck`
- `pnpm test`

## Deliverables

- approvals-related code changes
- shared contracts for trust levels, approval decisions, and action policy foundations
- minimal enforcement hooks in `apps/core` and/or `apps/ai`
- telemetry linkage needed for foundational traceability
- tests for foundational approval-policy behavior
- docs updates
- brief implementation notes in the PR body

## No-Touch Constraints

- Do not delete the legacy Python runtime
- Do not perform unrelated repo cleanup
- Do not rewrite adjacent subsystems unless strictly necessary for this issue
- Do not introduce placeholder approvals code without documenting why it exists
- Do not drift into full approval UI, high-autonomy enablement, MCP gateway implementation, or frontend operator console work
- Do not leave policy behavior implicit or prompt-only

## Acceptance Criteria

- `packages/approvals` exists as the canonical approvals package
- Shared approval and trust-ladder contracts are stable and importable
- Trust levels, approval requests, policy rules, and enforcement outcomes are explicitly represented
- Minimal enforcement hooks exist where needed for foundational approval-policy behavior
- Telemetry linkage exists where required for traceability
- Existing runtime behavior remains intact or improves without breakage

## Notes for Agent

This workstream depends on the architecture baseline from WS0, canonical config/dependency governance from WS1, telemetry foundation from WS2, identity/doctrine groundwork from WS3, memory hierarchy/gateway groundwork from WS4, workspace-state groundwork from WS5, skills-engine groundwork from WS6, and environment-twin groundwork from WS7. Read the GitHub issue first, then this task card, then inspect the repo. Extract `Base Branch` and `Branch` from this card. Create or switch to the work branch from the stated base branch only. Stay strictly within WS08 scope.
