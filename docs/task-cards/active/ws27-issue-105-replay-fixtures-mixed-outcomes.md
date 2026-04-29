# Agent Task Card

- Issue URL: https://github.com/Deveadra/versa/issues/105
- Issue: #105
- Issue Title: [Orchestrator][WS27] Replay fixtures for mixed run outcomes
- Parent Epic: #98
- Workstream: WS27

- Task Card ID: WS27-ISSUE-TBD
- Task Card Name: replay-fixtures-mixed-outcomes
- Task Card File Name: ws27-issue-tbd-replay-fixtures-mixed-outcomes.md
- Task Card Path: docs/task-cards/active/ws27-issue-tbd-replay-fixtures-mixed-outcomes.md

- Status: Draft
- Priority: Medium
- Agent Type: Roo

- Base Branch: main
- Branch: orchestrator/ws27-replay-fixtures-mixed-outcomes
- PR Title: orchestrator(ws27): add replay fixtures for mixed run outcomes

- Depends On: #98, WS17, WS18, WS19, WS20
- Blocks: reliable regression testing for non-happy-path execution

## Objective

Add replay fixtures for mixed orchestrator outcomes so Ultron can be tested against realistic success, partial, blocked, failed, and code/process skew states instead of only idealized flows.

## In Scope

- Define and add replay fixtures for:
  - success
  - partial
  - blocked
  - failed
  - merged but issue still open / code-process skew
- Reuse existing run/result/summary/workspace/blocker contracts where possible
- Add tests or harness utilities that consume these fixtures
- Document the meaning of each replay state and how to use the fixtures safely

## Out of Scope

- Broad self-improvement / Dream Mode work
- Full e2e orchestration suite beyond fixture/replay support
- CI coverage-threshold work
- New product features unrelated to replay confidence
- Legacy Python rewrite

## Files/Areas to Inspect First

- `apps/ai/src/`
- `apps/core/src/`
- `packages/shared/`
- `packages/workspaces/`
- `packages/memory/`
- `packages/telemetry/`
- `docs/task-cards/active/ws17-*`
- `docs/task-cards/active/ws18-*`
- `docs/task-cards/active/ws19-*`
- `docs/task-cards/active/ws20-*`
- existing test fixtures and artifact formats

## Required Approach

1. Inspect current repo state before editing.
2. Read the GitHub issue first, then this task card.
3. Use realistic artifact shapes from actual orchestrator outputs where available.
4. Keep fixtures explicit, documented, and reusable.
5. Include at least one real code/process skew scenario reflecting merged-code/open-issue mismatch.
6. Keep the PR bounded to replay confidence work.

## Required Validation

- `pnpm install`
- `pnpm lint`
- `pnpm typecheck`
- `pnpm test`
- targeted replay/fixture test command(s)

## Deliverables

- Replay fixtures for mixed outcomes
- Tests or harness support consuming those fixtures
- Documentation for replay-state meanings and usage

## No-Touch Constraints

- Do not delete the legacy Python runtime
- Do not perform unrelated repo cleanup
- Do not turn this into broad backlog automation or Dream Mode work
- Do not create vague fixtures disconnected from real orchestrator artifacts
- Do not drift into smoke-test or CI-threshold work beyond what is needed

## Acceptance Criteria

- Replay fixtures exist for success, partial, blocked, failed, and code/process skew outcomes.
- Fixtures align with real or contract-accurate orchestrator artifact shapes.
- Tests/harness utilities can consume the fixtures deterministically.
- Documentation explains what each replay state represents and when to use it.
- Existing behavior remains intact.

## Notes for Agent

The “merged but issue still open” state is especially valuable because it reflects real code/process skew already seen in the current project history. Read the GitHub issue first, then this task card, then inspect the repo. Rename this file to the canonical issue-numbered pattern after the GitHub issue is created.
