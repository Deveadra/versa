# Agent Task Card

- Issue URL: https://github.com/Deveadra/versa/issues/79
- Issue: #79
- Issue Title: [Orchestrator][WS14] Task-card generator and refresh workflow
- Parent Epic: #77
- Workstream: WS14

- Task Card ID: WS14-ISSUE79
- Task Card Name: task-card-generator
- Task Card File Name: ws14-issue-79-task-card-generator.md
- Task Card Path: docs/task-cards/active/ws14-issue-79-task-card-generator.md

- Status: Active
- Priority: High
- Agent Type: Roo

- Base Branch: main
- Branch: orchestrator/ws14-task-card-generator
- PR Title: orchestrator(ws14): add task-card generation and refresh workflow

- Depends On: #81

## Objective

Implement task-card generation and refresh behavior so Ultron can convert structured GitHub issue intake data into repo-local agent task cards following the established task-card standard.

## In Scope

- Add a task-card render contract
- Add deterministic task-card generation from normalized issue intake data
- Add task-card parsing/validation where useful
- Support canonical task-card filename/path generation
- Support refresh behavior for existing task cards
- Preserve manual notes during refresh unless explicitly configured otherwise
- Add tests for generated task-card output
- Add documentation describing generator behavior, limitations, and authority expectations

## Out of Scope

- GitHub issue creation or mutation
- Roo handoff generation
- Sandbox execution
- Result ingestion
- PR summary generation
- Blocker issue generation
- Legacy Python runtime deletion or rewrite
- Broad repo cleanup

## Files/Areas to Inspect First

- `docs/templates/agent-task-card.md`
- `docs/task-cards/active/`
- `docs/task-cards/archive/`
- `packages/shared/`
- `apps/core/src/`
- `apps/ai/src/`
- Any issue-intake files added by WS13
- `README.md`
- `docs/redesign/`

## Required Approach

1. Inspect current repo state before editing.
2. Read issue #79 first, then this task card.
3. Inspect the output/contracts from WS13 if present.
4. Use the existing task-card standard as the required output shape.
5. Make generation deterministic and reviewable.
6. Do not destroy manual notes during refresh unless the behavior is explicit and tested.
7. Keep this PR bounded to task-card generation and refresh.

## Required Validation

- `pnpm install`
- `pnpm lint`
- `pnpm typecheck`
- `pnpm test`

## Deliverables

- task-card generator
- task-card render contract
- task-card validation/parsing helpers where useful
- filename/path generation helpers
- refresh behavior
- tests for generated cards and naming conventions
- docs describing generator behavior

## No-Touch Constraints

- Do not delete or rewrite the legacy Python runtime
- Do not create or update GitHub issues in this workstream
- Do not implement Roo handoff generation here
- Do not implement sandbox execution or result ingestion
- Do not overwrite human-authored task-card notes without explicit tested behavior
- Do not perform unrelated cleanup

## Acceptance Criteria

- Given structured issue intake data, the generator produces a valid task card with required fields
- Generated filenames follow `wsXX-issue-<issue-number>-<short-kebab-name>.md`
- Generated task cards include objective, scope, validation, acceptance criteria, branch, PR title, no-touch constraints, and notes for agent
- Existing task cards can be refreshed without destroying manual notes unless explicitly requested
- Tests verify required fields and naming conventions

## Notes for Agent

This workstream turns the manual task-card writing process into a reproducible capability. It should not try to perform execution yet. The output is a repo-local execution packet that later Roo/Codex handoff logic can consume.
