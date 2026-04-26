# Agent Task Card

- Issue URL: <full github issue url>
- Issue: #<issue-number>
- Issue Title: <issue-title>
- Parent Epic: #<epic-number>
- Workstream: WS##

- Task Card ID: WS##-ISSUE<issue-number>
- Task Card Name: <short-kebab-name>
- Task Card File Name: ws##-issue-<issue-number>-<short-kebab-name>.md
- Task Card Path: docs/task-cards/active/ws##-issue-<issue-number>-<short-kebab-name>.md

- Status: Draft
- Priority: High
- Agent Type: Roo

- Base Branch: main
- Branch: <epic-topic>/ws##-<short-kebab-name>
- PR Title: <epic-topic>(ws##): <summary>

- Depends On: <issue numbers or none>
- Blocks: <issue numbers or none>

## Objective
<copy the Goal section from the issue>

## In Scope
- <explicit files/packages/apps allowed>
- <contracts/docs/tests to add>
- <minimal runtime wiring allowed>

## Out of Scope
- <clear no-touch areas>
- <legacy Python rewrite exclusions>
- <UI work if not relevant>
- <MCP work if not relevant>

## Files/Areas to Inspect First
- `README.md`
- `package.json`
- `pnpm-workspace.yaml`
- `turbo.json`
- `apps/core/...`
- `apps/ai/...`
- `apps/web/...`
- `packages/shared/...`
- `packages/config/...`
- `packages/database/...`
- any relevant `docs/adr` or `docs/redesign` files

## Required Approach
1. Inspect current repo state before editing.
2. Preserve working behavior unless the issue explicitly changes it.
3. Prefer additive implementation over broad refactor.
4. Keep contracts typed and importable.
5. Add or update tests for new behavior.
6. Add or update docs for the workstream.
7. Keep the PR bounded to this issue only.

## Required Validation
- `pnpm install`
- `pnpm lint`
- `pnpm typecheck`
- `pnpm test`
- any issue-specific validation steps

## Deliverables
- code changes
- tests
- docs
- brief implementation notes in PR body

## No-Touch Constraints
- do not delete the legacy Python runtime
- do not perform unrelated repo cleanup
- do not rewrite adjacent subsystems unless strictly necessary for this issue
- do not introduce placeholder code without documenting why it exists

## Acceptance Criteria
- <criterion 1>
- <criterion 2>
- <criterion 3>

## Notes for Agent
- Read the GitHub issue first.
- Read this task card second.
- Extract `Base Branch` and `Branch` from this card.
- Create/switch to the correct work branch from the stated base branch only.
- Stay strictly within scope.
