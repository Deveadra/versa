# Agent Task Card

- Issue URL: https://github.com/Deveadra/versa/issues/87
- Issue: #87
- Issue Title: [Orchestrator][WS19] Post-run workspace and memory update
- Parent Epic: #77
- Workstream: WS19

- Task Card ID: WS19-ISSUE-87
- Task Card Name: post-run-workspace-memory-update
- Task Card File Name: ws19-issue-87-post-run-workspace-memory-update.md
- Task Card Path: docs/task-cards/active/ws19-issue-87-post-run-workspace-memory-update.md

- Status: Active
- Priority: High
- Agent Type: Roo

- Base Branch: main
- Branch: orchestrator/ws19-post-run-workspace-memory-update
- PR Title: orchestrator(ws19): add post-run workspace and memory updates

- Depends On: #77, WS17, WS18
- Blocks: WS20, future lab loop / self-improvement work

## Objective

Implement post-run workspace and memory update behavior so each orchestrator execution attempt writes back durable state about what happened.

## In Scope

- Define a post-run workspace update contract
- Define a run history record model
- Represent issue/task-card/branch linkage
- Represent validation outcomes
- Represent blocker/follow-up linkage fields
- Represent important repo observations
- Add tests for post-run state update production
- Document what gets recorded after each run

## Out of Scope

- Full long-term memory implementation if not already available
- Creating follow-up GitHub issues
- Self-improvement / Dream Mode
- Broad analytics dashboards
- Deleting or rewriting the legacy Python runtime
- Unrelated repo cleanup

## Files/Areas to Inspect First

- `packages/workspaces/`
- `packages/memory/`
- `packages/shared/`
- `packages/logging/`
- `apps/core/src/`
- `apps/ai/src/`
- `docs/task-cards/active/ws17-issue-87-roo-dispatch-result-ingestion.md`
- `docs/task-cards/active/ws18-issue-87-result-summary-pr-packet.md`
- `docs/redesign/`
- `docs/adr/`

## Required Approach

1. Inspect current repo state before editing.
2. Read the GitHub issue first, then this task card.
3. Reuse WS17/WS18 result summary contracts where available.
4. Prefer additive storage hooks and typed records over broad memory rewrites.
5. Record failed, blocked, and partial runs honestly.
6. Keep storage behavior simple and testable.
7. Keep the PR bounded to WS19 only.

## Required Validation

- `pnpm install`
- `pnpm lint`
- `pnpm typecheck`
- `pnpm test`

## Deliverables

- Post-run workspace update contract
- Run history record model
- Issue/task-card/branch linkage model
- Validation outcome record
- Blocker/follow-up linkage fields
- Important repo observation fields
- Tests for post-run state record production
- Documentation describing post-run state updates

## No-Touch Constraints

- Do not delete the legacy Python runtime
- Do not perform unrelated repo cleanup
- Do not rewrite adjacent subsystems unless strictly necessary for this issue
- Do not silently expand task scope when a blocker is found
- Do not introduce placeholder code without documenting why it exists
- Do not drift into broad personal-assistant integrations, MCP expansion, or Dream Mode work

## Acceptance Criteria

- Given a result summary or PR review packet, Ultron can produce a durable post-run update record.
- The update records issue number, task-card path, branch, result status, validation pass/fail, blockers, follow-ups, and important repo observations.
- Records are structured enough for later retrieval and reporting.
- Failed or partial runs are recorded without pretending they succeeded.
- Tests cover successful, blocked, failed, and partial runs.
- Existing runtime behavior remains intact.

## Notes for Agent

This workstream is where Ultron starts accumulating operational memory. Keep the first version modest, durable, and honest. The goal is reliable state capture, not broad personality or Dream Mode behavior.

Read the GitHub issue first, then this task card, then inspect the repo. Extract `Base Branch` and `Branch` from this card. Create or switch to the work branch from the stated base branch only. Stay strictly within WS19 scope.
