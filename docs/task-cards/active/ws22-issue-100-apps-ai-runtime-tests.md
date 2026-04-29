# Agent Task Card

- Issue URL: https://github.com/Deveadra/versa/issues/100
- Issue: #100
- Issue Title: [Orchestrator][WS22] `apps/ai` runtime test coverage
- Parent Epic: #98
- Workstream: WS22

- Task Card ID: WS22-ISSUE-TBD
- Task Card Name: apps-ai-runtime-tests
- Task Card File Name: ws22-issue-tbd-apps-ai-runtime-tests.md
- Task Card Path: docs/task-cards/active/ws22-issue-tbd-apps-ai-runtime-tests.md

- Status: Draft
- Priority: Critical
- Agent Type: Roo

- Base Branch: main
- Branch: orchestrator/ws22-apps-ai-runtime-tests
- PR Title: orchestrator(ws22): add real runtime-facing tests for apps/ai

- Depends On: #98, WS17, WS18, WS19, WS20
- Blocks: trustworthy runtime control-plane evolution

## Objective

Add real runtime-facing tests for `apps/ai`, which is the thin but critical control-plane surface where Ultron’s future runtime behavior is expected to live.

## In Scope

- Inspect `apps/ai` current endpoints, runtime modules, and control-plane responsibilities
- Add tests for real behavior, not placeholder pass-through assertions
- Cover runtime endpoints and orchestration-facing flows such as planning, skills, workspace analysis, memory search, approval hooks, and health behavior where implemented
- Add focused mocks/fakes only at true external boundaries
- Add docs describing the test strategy for `apps/ai`

## Out of Scope

- Broad refactor of `apps/ai` architecture beyond what tests require
- New runtime feature implementation unrelated to testing confidence
- CI coverage-threshold work beyond local issue needs
- Full service composition/deployment smoke tests
- Legacy Python rewrite
- MCP policy work outside the `apps/ai` surface under test

## Files/Areas to Inspect First

- `apps/ai/src/server.ts`
- `apps/ai/src/`
- `packages/shared/`
- `packages/skills/`
- `packages/workspaces/`
- `packages/memory/`
- `packages/approvals/`
- `packages/telemetry/`
- existing test patterns in `apps/core/` and `packages/*`
- `docs/redesign/`

## Required Approach

1. Inspect current repo state before editing.
2. Read the GitHub issue first, then this task card.
3. Test `apps/ai` as a runtime boundary, not only as internal helper functions.
4. Prefer endpoint/module behavior tests that validate contracts and side effects.
5. Keep mocks at external seams only.
6. Make thin surfaces prove real behavior.
7. Keep the PR bounded to `apps/ai` testing confidence.

## Required Validation

- `pnpm install`
- `pnpm lint`
- `pnpm typecheck`
- `pnpm test`
- targeted `apps/ai` test command(s)

## Deliverables

- Real `apps/ai` runtime-facing tests
- Any minimal test utilities/fakes needed for control-plane boundaries
- Documentation for `apps/ai` test strategy
- Updated assertions for health/degradation behavior if present

## No-Touch Constraints

- Do not delete the legacy Python runtime
- Do not perform unrelated repo cleanup
- Do not turn this into a broad runtime rewrite
- Do not leave `apps/ai` with only thin smoke assertions and call it complete
- Do not drift into coverage-threshold or replay-fixture work unless strictly required

## Acceptance Criteria

- `apps/ai` has real tests covering meaningful runtime behavior.
- Critical routes/modules are exercised with contract-aware assertions.
- External dependencies are mocked/faked only at appropriate seams.
- Tests improve confidence in `apps/ai` as Ultron’s runtime/control plane.
- Existing behavior remains intact.

## Notes for Agent

This workstream matters because `apps/ai` is the future control-plane surface. Thin tests here would create false confidence. Read the GitHub issue first, then this task card, then inspect the repo. Rename this file to the canonical issue-numbered pattern after the GitHub issue is created.
