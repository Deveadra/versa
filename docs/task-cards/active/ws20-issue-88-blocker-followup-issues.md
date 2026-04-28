# Agent Task Card

- Issue URL: https://github.com/Deveadra/versa/issues/88
- Issue: #88
- Issue Title: [Orchestrator][WS20] Blocker and follow-up issue generation
- Parent Epic: #77
- Workstream: WS20

- Task Card ID: WS20-ISSUE-88
- Task Card Name: blocker-followup-issues
- Task Card File Name: ws20-issue-88-blocker-followup-issues.md
- Task Card Path: docs/task-cards/active/ws20-issue-88-blocker-followup-issues.md

- Status: Active
- Priority: High
- Agent Type: Roo

- Base Branch: main
- Branch: orchestrator/ws20-blocker-followup-issues
- PR Title: orchestrator(ws20): add blocker and follow-up issue generation

- Depends On: #77, WS17, WS18, WS19
- Blocks: future backlog automation and lab loop / self-improvement work

## Objective

Implement blocker and follow-up issue generation so Ultron can turn out-of-scope execution blockers into clear linked GitHub issue drafts.

## In Scope

- Define a blocker classification contract
- Define a follow-up issue draft contract
- Generate follow-up issue drafts from blocked/partial run results
- Model linkage back to parent issue, task card, branch, and run
- Create a safe issue body template
- Document optional GitHub issue creation boundary/design if appropriate
- Add tests for blocker classification and issue draft generation
- Document the blocker handling workflow

## Out of Scope

- Automatic GitHub issue creation without explicit approval path
- Broad backlog planning
- Self-improvement / Dream Mode
- Changing original task scope silently
- Deleting or rewriting the legacy Python runtime
- Unrelated repo cleanup

## Files/Areas to Inspect First

- `packages/shared/`
- `packages/workspaces/`
- `packages/logging/`
- `apps/core/src/`
- `apps/ai/src/`
- `docs/task-cards/active/ws17-issue-88-roo-dispatch-result-ingestion.md`
- `docs/task-cards/active/ws18-issue-88-result-summary-pr-packet.md`
- `docs/task-cards/active/ws19-issue-88-post-run-workspace-memory-update.md`
- `docs/redesign/`
- `docs/adr/`

## Required Approach

1. Inspect current repo state before editing.
2. Read the GitHub issue first, then this task card.
3. Reuse WS17/WS18/WS19 run, summary, and post-run contracts where available.
4. Classify blockers before generating issue drafts.
5. Generate drafts only unless an explicit approval/write path already exists.
6. Ensure blockers stay linked to their originating issue/task card/run.
7. Keep the PR bounded to WS20 only.

## Required Validation

- `pnpm install`
- `pnpm lint`
- `pnpm typecheck`
- `pnpm test`

## Deliverables

- Blocker classification contract
- Follow-up issue draft contract
- Generator from blocked/partial run results
- Parent issue/task-card/run linkage model
- Safe follow-up issue body template
- Tests for dependency, missing-contract, environment, validation, and scope-expansion blockers
- Documentation describing blocker handling workflow

## No-Touch Constraints

- Do not delete the legacy Python runtime
- Do not perform unrelated repo cleanup
- Do not rewrite adjacent subsystems unless strictly necessary for this issue
- Do not silently expand task scope when a blocker is found
- Do not introduce placeholder code without documenting why it exists
- Do not drift into broad personal-assistant integrations, MCP expansion, or Dream Mode work

## Acceptance Criteria

- Given a blocked or partial run result, Ultron can classify the blocker type.
- Ultron can generate a linked follow-up GitHub issue draft.
- Drafts include parent epic, originating issue, task card, branch/run context, blocker summary, suggested scope, acceptance criteria, and constraints.
- Out-of-scope blockers are not folded back into the original task silently.
- Tests cover dependency blockers, missing-contract blockers, environment blockers, validation blockers, and scope-expansion blockers.
- Existing runtime behavior remains intact.

## Notes for Agent

This workstream closes the loop between execution and backlog management. It should make blockers visible and actionable without giving agents permission to silently expand scope.

Read the GitHub issue first, then this task card, then inspect the repo. Extract `Base Branch` and `Branch` from this card. Create or switch to the work branch from the stated base branch only. Stay strictly within WS20 scope.
