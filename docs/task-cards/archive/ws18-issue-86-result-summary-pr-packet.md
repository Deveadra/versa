# Agent Task Card

- Issue URL: https://github.com/Deveadra/versa/issues/86
- Issue: #86
- Issue Title: [Orchestrator][WS18] Result summary and PR review packet
- Parent Epic: #77
- Workstream: WS18

- Task Card ID: WS18-ISSUE-86
- Task Card Name: result-summary-pr-packet
- Task Card File Name: ws18-issue-86-result-summary-pr-packet.md
- Task Card Path: docs/task-cards/active/ws18-issue-86-result-summary-pr-packet.md

- Status: Active
- Priority: High
- Agent Type: Roo

- Base Branch: main
- Branch: orchestrator/ws18-result-summary-pr-packet
- PR Title: orchestrator(ws18): add result summaries and PR review packets

- Depends On: #77, WS17

## Objective

Implement result summary and PR review packet generation so Ultron can turn an ingested execution result into a clear review-ready summary.

## In Scope

- Define a result summary contract
- Define a PR review packet contract
- Generate summaries from ingested run result data
- Model changed-files summaries
- Model validation summaries
- Add known-followups and risk/migration notes sections
- Add tests for generated summaries and PR packet output
- Document example PR review packet output

## Out of Scope

- Opening PRs
- Creating GitHub issues
- Post-run workspace/memory writeback beyond producing data for it
- Self-improvement / Dream Mode
- Deleting or rewriting the legacy Python runtime
- Unrelated repo cleanup

## Files/Areas to Inspect First

- `packages/shared/`
- `packages/workspaces/`
- `packages/logging/`
- `apps/core/src/`
- `apps/ai/src/`
- `docs/task-cards/active/ws17-issue-86-roo-dispatch-result-ingestion.md`
- `docs/redesign/`
- `docs/adr/`

## Required Approach

1. Inspect current repo state before editing.
2. Read the GitHub issue first, then this task card.
3. Reuse WS17 run/result contracts where available.
4. Keep output deterministic and reviewable.
5. Handle missing data explicitly instead of fabricating summaries.
6. Include enough structure for a future GitHub PR creation step to consume.
7. Keep the PR bounded to WS18 only.

## Required Validation

- `pnpm install`
- `pnpm lint`
- `pnpm typecheck`
- `pnpm test`

## Deliverables

- Result summary contract
- PR review packet contract
- Generator from ingested run result fixtures
- Changed-files summary model
- Validation summary model
- Known-followups and risk/migration notes sections
- Tests for success, partial, failed, and blocked run summaries
- Documentation with example PR review packet

## No-Touch Constraints

- Do not delete the legacy Python runtime
- Do not perform unrelated repo cleanup
- Do not rewrite adjacent subsystems unless strictly necessary for this issue
- Do not silently expand task scope when a blocker is found
- Do not introduce placeholder code without documenting why it exists
- Do not drift into broad personal-assistant integrations, MCP expansion, or Dream Mode work

## Acceptance Criteria

- Given an ingested run result, Ultron can generate a structured result summary.
- The summary includes status, issue, task card, branch, changed files, validation results, blockers, follow-ups, and review notes.
- The PR packet includes a usable PR title/body draft when enough data is available.
- Missing data is handled gracefully and flagged clearly.
- Tests cover success, partial, failed, and blocked run summaries.
- Existing runtime behavior remains intact.

## Notes for Agent

This workstream turns raw executor output into something a human reviewer can actually use. It should prepare the path for PR automation later without opening PRs itself.

Read the GitHub issue first, then this task card, then inspect the repo. Extract `Base Branch` and `Branch` from this card. Create or switch to the work branch from the stated base branch only. Stay strictly within WS18 scope.
