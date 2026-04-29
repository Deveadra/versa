# Agent Task Card

- Issue URL: https://github.com/Deveadra/versa/issues/101
- Issue: #101
- Issue Title: [Orchestrator][WS23] `packages/workspaces` confidence suite
- Parent Epic: #98
- Workstream: WS23

- Task Card ID: WS23-ISSUE-TBD
- Task Card Name: workspaces-confidence-suite
- Task Card File Name: ws23-issue-tbd-workspaces-confidence-suite.md
- Task Card Path: docs/task-cards/active/ws23-issue-tbd-workspaces-confidence-suite.md

- Status: Draft
- Priority: High
- Agent Type: Roo

- Base Branch: main
- Branch: orchestrator/ws23-workspaces-confidence-suite
- PR Title: orchestrator(ws23): strengthen packages/workspaces test confidence

- Depends On: #98, WS19
- Blocks: durable project-state trust and post-run state continuity

## Objective

Add a real confidence suite for `packages/workspaces` so durable project state sits on strong testing footing rather than relying on comparatively thin coverage.

## In Scope

- Add or strengthen tests for workspace creation, update, checkpointing, decisions, blockers, next actions, and context-bundle retrieval where implemented
- Cover durable state behavior and invariants important to Ultron’s ongoing project continuity
- Add tests for edge cases such as repeated updates, ordering, merge/appending rules, and missing workspace lookups where relevant
- Document the confidence goals and remaining gaps for workspace state

## Out of Scope

- Broad redesign of workspace data model unrelated to test confidence
- Memory boundary enforcement work
n- CI coverage-threshold work beyond what this issue directly needs
- New UI/dashboard work
- Legacy Python rewrite
- Broad orchestrator e2e work outside workspace-focused confidence

## Files/Areas to Inspect First

- `packages/workspaces/src/`
- `packages/shared/`
- `packages/database/`
- `apps/core/src/`
- `apps/ai/src/`
- `docs/task-cards/active/ws19-*`
- existing workspace tests and fixtures
- `docs/redesign/`

## Required Approach

1. Inspect current repo state before editing.
2. Read the GitHub issue first, then this task card.
3. Treat workspace state as durable operating context, not incidental metadata.
4. Add tests around behaviors and invariants that matter to Ultron’s continuity.
5. Prefer realistic fixtures and repository/service-level assertions where appropriate.
6. Keep the PR bounded to workspace confidence work.

## Required Validation

- `pnpm install`
- `pnpm lint`
- `pnpm typecheck`
- `pnpm test`
- targeted `packages/workspaces` test command(s)

## Deliverables

- Expanded workspace confidence test suite
- Any supporting workspace fixtures/utilities
- Documentation describing workspace confidence goals and open gaps

## No-Touch Constraints

- Do not delete the legacy Python runtime
- Do not perform unrelated repo cleanup
- Do not refactor workspace semantics broadly unless strictly necessary for this issue
- Do not drift into memory-boundary enforcement or replay-fixture work

## Acceptance Criteria

- `packages/workspaces` has meaningful tests for durable project-state behaviors.
- Tests cover checkpoints, decisions, blockers, next actions, and context retrieval where implemented.
- Edge cases important to long-lived state are exercised.
- Existing behavior remains intact.

## Notes for Agent

Workspace state is one of the highest-ROI features in the redesign. It should not sit below the confidence level of less central packages. Read the GitHub issue first, then this task card, then inspect the repo. Rename this file to the canonical issue-numbered pattern after the GitHub issue is created.
