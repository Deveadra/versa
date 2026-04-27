# Agent Task Card

- Issue URL: https://github.com/Deveadra/versa/issues/51
- Issue: #51
- Issue Title: [Redesign][WS6] Skills and procedural execution engine
- Parent Epic: #42
- Workstream: WS06

- Task Card ID: WS06-ISSUE51
- Task Card Name: skills-engine
- Task Card File Name: ws06-issue-51-skills-engine.md
- Task Card Path: docs/task-cards/active/ws06-issue-51-skills-engine.md

- Status: Active
- Priority: High
- Agent Type: Roo

- Base Branch: main
- Branch: redesign/ws06-skills-engine
- PR Title: redesign(ws06): implement skills and procedural execution engine

- Depends On: #43, #46, #47, #48, #49, #50
- Blocks: #52, #53, #54, #55, #56, #57

## Objective

Create reusable bounded skills for operational workflows.

## In Scope

- Create `packages/skills` as the canonical skills package
- Add typed shared contracts in `packages/shared` for:
  - skill definition
  - skill metadata
  - skill inputs and outputs
  - skill execution requests
  - skill execution results
  - skill validation requirements
  - skill failure and status reporting
- Add minimal execution hooks in `apps/ai` and/or `apps/core` only where needed to establish the foundational skill execution path
- Add docs describing:
  - what a skill is
  - how skills are defined
  - how skills are executed
  - how skills remain bounded and reusable
  - how skills differ from broad autonomous behavior
- Add tests for core skill contracts and foundational execution behavior
- Keep implementation additive and bounded to skills/procedural execution foundations only

## Out of Scope

- Self-improvement lab
- Destructive autonomous behaviors
- MCP gateway implementation
- Broad runtime rewrites unrelated to skill foundations
- Full orchestration system redesign beyond minimal skill hooks
- Unrelated cleanup

## Files/Areas to Inspect First

- `packages/skills/`
- `packages/shared/`
- `apps/ai/`
- `apps/core/`
- `README.md`
- `docs/adr/`
- `docs/redesign/`
- `docs/task-cards/active/ws00-issue-43-architecture-baseline.md`
- `docs/task-cards/active/ws01-issue-46-config-governance.md`
- `docs/task-cards/active/ws02-issue-47-telemetry-foundation.md`
- `docs/task-cards/active/ws03-issue-48-identity-doctrine.md`
- `docs/task-cards/active/ws04-issue-49-memory-gateway.md`
- `docs/task-cards/active/ws05-issue-50-workspace-state.md`

## Required Approach

1. Inspect current repo state before editing.
2. Read issue #51 first, then this task card.
3. Preserve working behavior unless the issue explicitly changes it.
4. Prefer additive contracts, package scaffolding, execution hooks, docs, and tests over broad refactors.
5. Keep skills bounded, reusable, and explicitly typed.
6. Do not let “skills” become vague open-ended autonomous behavior.
7. Keep the PR bounded to this issue only.

## Required Validation

- `pnpm install`
- `pnpm lint`
- `pnpm typecheck`
- `pnpm test`

## Deliverables

- skills-related code changes
- shared contracts for skill definitions and foundational execution behavior
- minimal execution hooks in `apps/ai` and/or `apps/core`
- tests for foundational skill behavior
- docs updates
- brief implementation notes in the PR body

## No-Touch Constraints

- Do not delete the legacy Python runtime
- Do not perform unrelated repo cleanup
- Do not rewrite adjacent subsystems unless strictly necessary for this issue
- Do not introduce placeholder skills code without documenting why it exists
- Do not drift into self-improvement lab work, broad autonomy features, MCP gateway implementation, or frontend operator console work
- Do not implement destructive or unsafely autonomous behaviors in this issue

## Acceptance Criteria

- `packages/skills` exists as the canonical skills package
- Shared skills contracts are stable and importable
- Skills are represented as bounded, reusable execution units
- Minimal execution hooks exist where needed for foundational skill invocation
- Tests cover the foundational skill path being introduced
- Existing runtime behavior remains intact or improves without breakage

## Notes for Agent

This workstream depends on the architecture baseline from WS0, canonical config/dependency governance from WS1, telemetry foundation from WS2, identity/doctrine groundwork from WS3, memory hierarchy/gateway groundwork from WS4, and workspace-state groundwork from WS5. Read the GitHub issue first, then this task card, then inspect the repo. Extract `Base Branch` and `Branch` from this card. Create or switch to the work branch from the stated base branch only. Stay strictly within WS06 scope.
