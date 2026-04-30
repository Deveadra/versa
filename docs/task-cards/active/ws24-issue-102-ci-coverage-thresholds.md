# Agent Task Card

- Issue URL: https://github.com/Deveadra/versa/issues/102
- Issue: #102
- Issue Title: [Orchestrator][WS24] CI coverage reporting and thresholds
- Parent Epic: #98
- Workstream: WS24

- Task Card ID: WS24-ISSUE-102
- Task Card Name: ci-coverage-thresholds
- Task Card File Name: ws24-issue-102-ci-coverage-thresholds.md
- Task Card Path: docs/task-cards/active/ws24-issue-102-ci-coverage-thresholds.md

- Status: Draft
- Priority: High
- Agent Type: Roo

- Base Branch: main
- Branch: orchestrator/ws24-ci-coverage-thresholds
- PR Title: orchestrator(ws24): publish coverage metrics and enforce thresholds

- Depends On: #98, WS21, WS22, WS23
- Blocks: reliable CI confidence signals

## Objective

Publish coverage metrics in CI for both TypeScript and Python and set explicit minimum thresholds for the packages/apps that matter most to Ultron’s runtime confidence.

## In Scope

- Inspect existing CI workflows, test commands, and coverage capabilities
- Add coverage publication/reporting for TypeScript
- Add coverage publication/reporting for Python where feasible in the current repo setup
- Set explicit minimum thresholds for critical areas such as:
  - `packages/integrations`
  - `packages/memory`
  - `packages/workspaces`
  - `packages/approvals`
  - `packages/environment`
  - `apps/ai`
- Document what the thresholds are and how to update them safely

## Out of Scope

- Inflating thresholds unrealistically without runtime justification
- Massive CI redesign unrelated to coverage publication and thresholds
- Rewriting test frameworks
- Creating entirely new feature tests that belong to other workstreams
- Legacy Python runtime rewrite

## Files/Areas to Inspect First

- `.github/workflows/`
- `package.json`
- `pnpm-workspace.yaml`
- `turbo.json`
- `pyproject.toml`
- existing test configs for TS and Python
- `packages/integrations/`
- `packages/memory/`
- `packages/workspaces/`
- `packages/approvals/`
- `packages/environment/`
- `apps/ai/`

## Required Approach

1. Inspect current repo state before editing.
2. Read the GitHub issue first, then this task card.
3. Treat “green CI” as insufficient unless it surfaces meaningful coverage confidence.
4. Choose threshold levels that are explicit, documented, and supportable.
5. Keep the PR bounded to coverage reporting and threshold enforcement.
6. Document how failures should be interpreted and fixed.

## Required Validation

- `pnpm install`
- `pnpm lint`
- `pnpm typecheck`
- `pnpm test`
- relevant coverage commands for TS and Python
- CI workflow validation as appropriate

## Deliverables

- Coverage reporting in CI for TS and Python
- Explicit minimum thresholds for critical packages/apps
- Docs describing current thresholds and maintenance expectations
- Any small workflow/config changes needed to enforce them

## No-Touch Constraints

- Do not delete the legacy Python runtime
- Do not perform unrelated repo cleanup
- Do not hide low coverage by excluding critical paths without justification
- Do not create misleading thresholds that provide false confidence
- Do not drift into service smoke tests or replay-fixture work

## Acceptance Criteria

- CI publishes coverage metrics for both TypeScript and Python.
- Critical packages/apps have explicit minimum thresholds.
- Threshold failures are visible and actionable.
- Documentation explains what is covered, what thresholds exist, and how to update them.
- Existing runtime behavior remains intact.

## Notes for Agent

This workstream makes “green” mean more than “the commands returned zero.” Keep it grounded and honest. Read the GitHub issue first, then this task card, then inspect the repo. Rename this file to the canonical issue-numbered pattern after the GitHub issue is created.
