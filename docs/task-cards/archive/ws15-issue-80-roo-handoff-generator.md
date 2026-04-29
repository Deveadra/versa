# Agent Task Card

- Issue URL: https://github.com/Deveadra/versa/issues/80
- Issue: #80
- Issue Title: [Orchestrator][WS15] Roo executor handoff generator
- Parent Epic: #77
- Workstream: WS15

- Task Card ID: WS15-ISSUE80
- Task Card Name: roo-handoff-generator
- Task Card File Name: ws15-issue-80-roo-handoff-generator.md
- Task Card Path: docs/task-cards/active/ws15-issue-80-roo-handoff-generator.md

- Status: Active
- Priority: High
- Agent Type: Roo

- Base Branch: main
- Branch: orchestrator/ws15-roo-handoff-generator
- PR Title: orchestrator(ws15): add Roo executor handoff generation

- Depends On: #81, #82

## Objective

Implement the Roo executor handoff generator so Ultron can convert a GitHub issue plus active task card into a precise Roo-ready execution prompt.

## In Scope

- Add a Roo handoff contract
- Render a Roo-ready handoff from issue intake data and task-card data
- Include required handoff sections:
  - authority order
  - issue URL/number/title
  - task-card path
  - base branch and target branch
  - objective
  - in-scope work
  - out-of-scope work
  - files/areas to inspect first
  - required validation
  - no-touch constraints
  - expected deliverables
  - blocker reporting rules
  - expected final response format
- Add branch/base branch extraction support
- Add tests for generated handoff content
- Add docs with an example generated handoff

## Out of Scope

- Live Roo dispatch
- Command execution
- Sandbox worktree creation
- Result ingestion
- PR summary generation
- Blocker issue creation
- Legacy Python runtime deletion or rewrite

## Files/Areas to Inspect First

- Any issue-intake files from WS13
- Any task-card generator files from WS14
- `docs/templates/agent-task-card.md`
- `docs/task-cards/active/`
- `packages/shared/`
- `apps/core/src/`
- `apps/ai/src/`
- `docs/redesign/`
- `README.md`

## Required Approach

1. Inspect current repo state before editing.
2. Read issue #80 first, then this task card.
3. Consume structured issue/task-card data where available.
4. Keep generated handoffs deterministic and reviewable.
5. Make the handoff clear enough for Roo to execute without guessing.
6. Include blocker reporting instructions so Roo does not expand scope silently.
7. Keep this PR bounded to handoff generation only.

## Required Validation

- `pnpm install`
- `pnpm lint`
- `pnpm typecheck`
- `pnpm test`

## Deliverables

- Roo handoff contract
- handoff renderer
- branch/base branch extraction support
- blocker-reporting handoff instructions
- tests for required prompt sections
- docs with example generated handoff

## No-Touch Constraints

- Do not delete or rewrite the legacy Python runtime
- Do not implement actual Roo dispatch in this workstream
- Do not execute commands or modify branches in this workstream
- Do not generate task cards here beyond consuming their data
- Do not implement sandbox execution or result ingestion
- Do not perform unrelated cleanup

## Acceptance Criteria

- Given normalized issue data and a task card, the generator produces a Roo-ready handoff prompt
- The handoff includes issue URL/number, task-card path, branch/base branch, objective, scope, validation, no-touch constraints, and expected response format
- The handoff instructs Roo to inspect current repo state before editing
- The handoff instructs Roo to report blockers rather than expanding scope
- Tests verify required sections are present

## Notes for Agent

This card creates the bridge between planning and execution. The generated handoff should be usable directly in Roo, but this issue should not attempt to automate Roo itself.
