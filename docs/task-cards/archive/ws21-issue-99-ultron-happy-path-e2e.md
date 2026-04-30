# Agent Task Card

- Issue URL: https://github.com/Deveadra/versa/issues/99
- Issue: #99
- Issue Title: [Orchestrator][WS21] End-to-end Ultron happy path
- Parent Epic: #98
- Workstream: WS21

- Task Card ID: WS21-ISSUE-TBD
- Task Card Name: ultron-happy-path-e2e
- Task Card File Name: ws21-issue-tbd-ultron-happy-path-e2e.md
- Task Card Path: docs/task-cards/active/ws21-issue-tbd-ultron-happy-path-e2e.md

- Status: Draft
- Priority: Critical
- Agent Type: Roo

- Base Branch: main
- Branch: orchestrator/ws21-ultron-happy-path-e2e
- PR Title: orchestrator(ws21): add end-to-end Ultron happy-path test suite

- Depends On: #98, #77, WS13, WS14, WS15, WS16, WS17, WS18, WS19, WS20
- Blocks: high-confidence runtime rollout and future orchestrator expansion

## Objective

Implement a true end-to-end happy-path test suite that exercises Ultron’s full issue-to-execution automation chain from issue intake through post-run updates and blocker/follow-up draft generation.

## In Scope

- Add an orchestrator-level end-to-end test harness
- Exercise this chain in one bounded happy-path flow:
  - issue intake
  - task card generation/refresh
  - Roo handoff generation
  - sandbox preparation
  - result ingestion
  - result summary / PR packet generation
  - post-run workspace/memory update
  - blocker/follow-up draft generation path where applicable
- Reuse real WS13–WS20 contracts and artifacts where possible
- Add fixtures or seeded inputs needed to make the flow deterministic
- Add docs describing what the happy-path suite proves and what it still does not prove

## Out of Scope

- Broad new orchestrator features unrelated to the happy-path test chain
- Rewriting existing runtime modules just to simplify testing
- New product integrations
- Legacy Python runtime rewrite
- Broad CI overhaul beyond what this test suite strictly needs
- Service boot/deployment smoke work outside this issue

## Files/Areas to Inspect First

- `apps/ai/src/`
- `apps/core/src/`
- `packages/shared/`
- `packages/workspaces/`
- `packages/memory/`
- `packages/approvals/`
- `packages/integrations/`
- `docs/task-cards/active/ws13-*`
- `docs/task-cards/active/ws14-*`
- `docs/task-cards/active/ws15-*`
- `docs/task-cards/active/ws16-*`
- `docs/task-cards/active/ws17-*`
- `docs/task-cards/active/ws18-*`
- `docs/task-cards/active/ws19-*`
- `docs/task-cards/active/ws20-*`
- `docs/orchestrator/`
- existing test directories across `apps/` and `packages/`

## Required Approach

1. Inspect current repo state before editing.
2. Read the GitHub issue first, then this task card.
3. Treat the happy-path suite as a real chain test, not a collection of isolated unit mocks.
4. Prefer deterministic fixtures and narrow seams over brittle deep mocks.
5. Reuse existing run/result/summary/workspace contracts from WS13–WS20.
6. Keep the PR bounded to the end-to-end runtime confidence slice.
7. Document precisely which boundaries are real and which are simulated.

## Required Validation

- `pnpm install`
- `pnpm lint`
- `pnpm typecheck`
- `pnpm test`
- targeted orchestrator e2e test command(s)

## Deliverables

- End-to-end happy-path test harness
- Deterministic fixtures / seeded inputs
- Assertions covering full chain outputs and state transitions
- Documentation for test scope and limitations
- Any minimal test utilities needed to support the suite

## No-Touch Constraints

- Do not delete the legacy Python runtime
- Do not perform unrelated repo cleanup
- Do not rewrite adjacent subsystems unless strictly necessary for this issue
- Do not replace real contracts with ad hoc test-only shapes
- Do not silently narrow the chain so the “end-to-end” label becomes misleading
- Do not drift into CI coverage-threshold work or smoke-test work beyond this issue

## Acceptance Criteria

- One happy-path suite exercises the full issue-to-execution chain end to end.
- The suite proves state continuity across intake, handoff, execution result ingestion, summary generation, and post-run updates.
- Assertions verify generated artifacts and durable state changes at the correct boundaries.
- The suite is deterministic enough for CI use.
- Documentation explains what is real, what is simulated, and what follow-up testing still exists.
- Existing runtime behavior remains intact.

## Notes for Agent

This is the most important confidence test in the current orchestrator phase. It should prove the chain, not just the pieces. Read the GitHub issue first, then this task card, then inspect the repo. Rename this file to the canonical issue-numbered pattern after the GitHub issue is created.
