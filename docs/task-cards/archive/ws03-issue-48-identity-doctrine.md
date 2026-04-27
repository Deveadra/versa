# Agent Task Card

- Issue URL: https://github.com/Deveadra/versa/issues/48
- Issue: #48
- Issue Title: [Redesign][WS3] Identity and doctrine subsystem
- Parent Epic: #42
- Workstream: WS03

- Task Card ID: WS03-ISSUE48
- Task Card Name: identity-doctrine
- Task Card File Name: ws03-issue-48-identity-doctrine.md
- Task Card Path: docs/task-cards/active/ws03-issue-48-identity-doctrine.md

- Status: Active
- Priority: High
- Agent Type: Roo

- Base Branch: main
- Branch: redesign/ws03-identity-doctrine
- PR Title: redesign(ws03): add identity and doctrine subsystem

- Depends On: #43, #46
- Blocks: #49, #50, #51, #52, #53, #54, #55, #56, #57

## Objective

Create the identity and doctrine subsystem that defines who Aerith/Ultron is operationally and what principles govern behavior.

## In Scope

- Create one canonical doctrine package:
  - prefer `packages/doctrine`
  - do not create both `packages/doctrine` and `packages/identity`
- Add typed shared contracts in `packages/shared` for doctrine and identity-related structures
- Add related docs describing:
  - doctrine purpose
  - doctrine ownership
  - doctrine loading/retrieval model
  - doctrine versioning expectations
- Add minimal config additions only if required for doctrine package loading or feature gating
- Keep implementation additive and bounded to doctrine/identity foundation work only

## Out of Scope

- Full prompt overhaul
- Legacy Python personality rewrite
- Self-improvement or lab-loop changes
- Telemetry implementation beyond any already-defined contract touchpoints
- Broad runtime rewrites unrelated to doctrine/identity foundations
- Unrelated cleanup

## Files/Areas to Inspect First

- `packages/shared/`
- `packages/config/`
- `apps/ai/`
- `apps/core/`
- `README.md`
- `docs/adr/`
- `docs/redesign/`
- `docs/task-cards/active/ws00-issue-43-architecture-baseline.md`
- `docs/task-cards/active/ws01-issue-46-config-governance.md`
- `docs/task-cards/active/ws02-issue-47-telemetry-foundation.md`

## Required Approach

1. Inspect current repo state before editing.
2. Read issue #48 first, then this task card.
3. Preserve working behavior unless the issue explicitly changes it.
4. Prefer additive contracts, package scaffolding, and docs over broad refactors.
5. Keep doctrine contracts typed and importable.
6. If a package name decision is needed, choose one canonical package and document the choice.
7. Keep the PR bounded to this issue only.

## Required Validation

- `pnpm install`
- `pnpm lint`
- `pnpm typecheck`
- `pnpm test`

## Deliverables

- doctrine-related code changes
- shared contracts for doctrine/identity foundation
- docs updates
- brief implementation notes in the PR body

## No-Touch Constraints

- Do not delete the legacy Python runtime
- Do not perform unrelated repo cleanup
- Do not rewrite adjacent subsystems unless strictly necessary for this issue
- Do not introduce placeholder doctrine code without documenting why it exists
- Do not drift into memory hierarchy, skills, environment twin, approvals, MCP gateway, or frontend operator console work

## Acceptance Criteria

- A single canonical doctrine package exists and is clearly named
- Doctrine and identity contracts are stable and importable
- Doctrine ownership, purpose, and versioning expectations are documented
- Any required config additions remain minimal and bounded
- Existing runtime behavior remains intact or improves without breakage

## Notes for Agent

This workstream depends on the architecture baseline from WS0 and canonical config/dependency governance from WS1. Read the GitHub issue first, then this task card, then inspect the repo. Extract `Base Branch` and `Branch` from this card. Create or switch to the work branch from the stated base branch only. Stay strictly within WS03 scope.
